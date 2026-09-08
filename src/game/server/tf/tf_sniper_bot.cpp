#include "cbase.h"
#include "tf_sniper_bot.h"

#include "player.h"
#include "tf_player.h"
#include "tf_bot_temp.h"
#include "tf_weapon_sniperrifle.h"
#include "tf_playerclass_shared.h"
#include "tf_gamerules.h"
#include "in_buttons.h"
#include "movehelper_server.h"
#include "datacache/imdlcache.h"

#include "tf_sniper_policy.h"

// Round/respawn/heal flow is already owned by round_manager.nut (see
// game/mod_tf/scripts/vscripts/) -- this file only drives movement/aim/
// scope/fire for the RL bot while it's alive. It does not touch team score,
// round counting, or respawn timing.
//
// 2026-08-25: dropped self-play (mirrored dual-role training against a
// frozen copy of itself) for training against a scripted opponent whose
// difficulty ramps over the run -- see sniper_duel_env.py's "scripted
// opponent" block comment for why. There is now exactly one trained role
// (RED, canonical frame, matches SELF_SPAWN/SELF_SPAWN_ANGLE) and no BLU-side
// RL bot, so the observation/action mirroring and the two-bot duel path this
// file used to have are both gone. Join BLU as a human to fight it.

// Real time isn't the same axis the toy env trained on (each training env
// "step" was an abstract unit, not a fixed wall-clock duration), so these
// two constants are our own real-time approximations of that env's per-step
// turn amount and episode length -- untested in-game, expect to retune once
// we can see the bot move.
static const float TURN_RATE_DEG_PER_SEC = 300.0f;
static const float EPISODE_DURATION_SECONDS = 20.0f;

// Must match sniper_duel_env.py's POSITION_LOW/POSITION_HIGH exactly -- see
// that file's comment for the full derivation. Surveyed from the compiled
// map's actual wall brushes (inner faces at x=-639/647, y=-479/459) minus a
// real player's ~24-unit collision hull radius, confirmed empirically via
// sniperbot_debug (both bots got stuck sliding along a wall at exactly
// wall_face -+ ~24.3). The toy env clips position to this box every step --
// the trained policy never saw an observation outside it.
static const float POSITION_LOW_X = -610.0f;
static const float POSITION_HIGH_X = 615.0f;
static const float POSITION_LOW_Y = -450.0f;
static const float POSITION_HIGH_Y = 430.0f;

struct SniperBotSlot_t
{
	CHandle<CTFPlayer> hBot;
	float flAliveSince;   // gpGlobals->curtime this life started, for the time_left approximation
	bool bWasAlive;
};

static SniperBotSlot_t g_SniperBot; // RED only -- see the file-header comment.

// Must match sniper_duel_env.py's SPAWN_JITTER exactly -- training added
// this same +-40 unit random offset to the spawn every episode, so the
// policy actually expects some position variety, not the exact spawn origin.
static const float SPAWN_JITTER = 40.0f;

// TF2's own spawn-point selection (run by ForceRespawn()) is what
// round_manager.nut's ResetRound() explicitly works around every round --
// it doesn't trust the engine to land a player back on "spawn_red" and
// instead looks the named entity up and teleports there directly. Do the
// same thing here for the same reason: whatever the engine's spawn-point
// resolution is actually doing (only one candidate point exists per team,
// per the .vmf, so this shouldn't be ambiguous, but evidently something
// about it isn't reliable for a freshly-created fake client), pin the
// position ourselves instead of trusting it.
//
// Also jitters x/y (not z, not facing angle -- training never varied those
// either) so the bot doesn't tele to the exact same spot life after life.
static void PinToNamedSpawn( CTFPlayer *pBot )
{
	CBaseEntity *pSpawn = gEntList.FindEntityByName( NULL, "spawn_red" );
	if ( pSpawn )
	{
		Vector vecSpawn = pSpawn->GetAbsOrigin();
		vecSpawn.x += RandomFloat( -SPAWN_JITTER, SPAWN_JITTER );
		vecSpawn.y += RandomFloat( -SPAWN_JITTER, SPAWN_JITTER );
		pBot->Teleport( &vecSpawn, &pSpawn->GetAbsAngles(), NULL );
	}
}

static CTFPlayer *SpawnOneSniperBot( const char *pszName )
{
	CBasePlayer *pPlayer = BotPutInServer( false, false, TF_TEAM_RED, TF_CLASS_SNIPER, pszName );
	if ( !pPlayer )
		return NULL;

	CTFPlayer *pBot = ToTFPlayer( pPlayer );
	pBot->SetPlayerType( CTFPlayer::RL_BOT );

	pBot->HandleCommand_JoinTeam( "red" );
	pBot->HandleCommand_JoinClass( GetPlayerClassData( TF_CLASS_SNIPER )->m_szClassName );
	pBot->ForceRespawn();
	PinToNamedSpawn( pBot );

	return pBot;
}

// Spawns (or respawns) the single RED RL bot. SniperBot_RunAll() picks up
// whichever live human is on BLU as its opponent -- see FindHumanOpponent.
void SniperBot_SpawnSolo()
{
	if ( g_SniperBot.hBot.Get() )
	{
		g_SniperBot.hBot->ForceRespawn();
		PinToNamedSpawn( g_SniperBot.hBot );
	}
	else
	{
		g_SniperBot.hBot = SpawnOneSniperBot( "jerry" );
	}
	g_SniperBot.flAliveSince = gpGlobals->curtime;
	g_SniperBot.bWasAlive = true;
}

void SniperBot_RemoveDuel()
{
	CTFPlayer *pBot = g_SniperBot.hBot.Get();
	if ( pBot )
	{
		engine->ServerCommand( UTIL_VarArgs( "kickid %d\n", pBot->GetUserID() ) );
	}
	g_SniperBot.hBot = NULL;
}

// Builds the observation vector in the exact order tf_sniper_policy_weights.h
// documents (== export_policy.py's OBS_KEY_ORDER), from the bot's point of
// view with pOpponent as "the opponent". No mirroring -- the bot only ever
// plays the canonical RED role (spawn_red, facing +x) the policy was trained
// as, so its real coordinates/yaw already match the training frame directly.
static void BuildObservation( CTFPlayer *pBot, CTFPlayer *pOpponent, float flAliveSince, float obs[SniperPolicy::kObsSize] )
{
	bool bOpponentVisible = pBot->FVisible( pOpponent );

	const Vector &vecSelf = pBot->GetAbsOrigin();
	const Vector &vecOpponent = pOpponent->GetAbsOrigin();

	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	float flYaw = AngleNormalize( pBot->EyeAngles().y );

	float flTimeLeft = 1.0f - ( ( gpGlobals->curtime - flAliveSince ) / EPISODE_DURATION_SECONDS );
	flTimeLeft = clamp( flTimeLeft, 0.0f, 1.0f );

	int i = 0;
	obs[i++] = bOpponentVisible ? vecOpponent.x : 0.0f;
	obs[i++] = bOpponentVisible ? vecOpponent.y : 0.0f;
	obs[i++] = bOpponentVisible ? 1.0f : 0.0f;
	obs[i++] = ( pRifle && pRifle->IsZoomed() ) ? 1.0f : 0.0f;
	obs[i++] = pRifle ? pRifle->GetProgress() : 0.0f;
	obs[i++] = flYaw;
	obs[i++] = clamp( (float)pBot->GetHealth() / (float)pBot->GetMaxHealth(), 0.0f, 1.0f );
	obs[i++] = vecSelf.x;
	obs[i++] = vecSelf.y;
	obs[i++] = flTimeLeft;
	Assert( i == SniperPolicy::kObsSize );
}

// Turns the policy's 5-float action (strafe, forward/back, turn, scope,
// fire -- see sniper_duel_env.py's action_space comment) into a CUserCmd
// and runs it, the same way tf_bot_temp.cpp's RunPlayerMove() does for the
// waypoint bots.
//
// 2026-08-25: the fire button is hard-gated on pBot->FVisible( pOpponent )
// here, regardless of what the policy outputs. Live testing showed the bot
// holding fire essentially constantly, including through cover -- reward
// shaping aimed at teaching this in training (a heavier penalty for firing
// with no LOS) was tried and reverted after it destabilized training (see
// sniper_duel_env.py's MISS_PENALTY comment); enforcing it as a real
// engine-visibility check here instead is simpler and can't be wrong the
// way a learned habit can.
//
// 2026-09-08: same story, second exploit. Live testing also showed the bot
// spinning continuously at max turn rate while holding the fire button down
// -- a fast spin sweeps across FVisible() often enough to intermittently
// land free kills without the policy ever needing to actually settle onto
// target. Four different attempts to close this in TRAINING itself (gating
// the agent's own fire on N consecutive on-target frames, a decaying
// version of the same, one with an added reward gradient, and finally an
// instantaneous check on the shooter's current turn action -- see
// sniper_duel_env.py's long comment above _resolve_fire) all destabilized
// PPO's training dynamics once tested against the real curriculum's actual
// shape, despite the underlying reward economy otherwise being proven
// stable. Enforcing "don't fire while turning fast" here instead, on the
// trained policy's raw action output, sidesteps training entirely -- same
// fix category as the FVisible() gate above, and the threshold matches what
// sniper_duel_env.py's abandoned MAX_TURN_WHILE_FIRING settled on before
// that whole mechanic was reverted.
static const float MAX_TURN_ACTION_WHILE_FIRING = 0.35f;

static void ApplyAction( CTFPlayer *pBot, CTFPlayer *pOpponent, const float action[SniperPolicy::kActionSize] )
{
	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	// Clamp frametime for the turn integration -- a single unusually long
	// server frame (e.g. a hitch right at bot spawn) would otherwise translate
	// into one giant snap-turn instead of a normal small per-tick step.
	float flTurnFrametime = MIN( gpGlobals->frametime, 0.1f );

	QAngle angViewAngles = pBot->EyeAngles();
	angViewAngles.y = AngleNormalize( angViewAngles.y + action[2] * TURN_RATE_DEG_PER_SEC * flTurnFrametime );
	angViewAngles.x = 0.0f;
	angViewAngles.z = 0.0f;

	unsigned short usButtons = 0;

	//only pulse the button on the tick our desired scope state
	// (from the policy) differs from the weapon's actual current state.
	bool bWantScope = action[3] > 0.0f;
	bool bIsZoomed = pRifle && pRifle->IsZoomed();
	if ( pRifle && bWantScope != bIsZoomed )
	{
		usButtons |= IN_ATTACK2;
	}

	if ( action[4] > 0.0f && pBot->FVisible( pOpponent ) && fabsf( action[2] ) <= MAX_TURN_ACTION_WHILE_FIRING )
	{
		usButtons |= IN_ATTACK;
	}

	CUserCmd cmd;
	Q_memset( &cmd, 0, sizeof( cmd ) );
	VectorCopy( angViewAngles, cmd.viewangles );
	cmd.forwardmove = action[1] * pBot->MaxSpeed();
	cmd.sidemove = action[0] * pBot->MaxSpeed();
	cmd.upmove = 0;
	cmd.buttons = usButtons;
	cmd.impulse = 0;
	cmd.random_seed = RandomInt( 0, 0x7fffffff );
	cmd.server_random_seed = cmd.random_seed;

	pBot->SetTimeBase( gpGlobals->curtime );

	MoveHelperServer()->SetHost( pBot );
	pBot->PlayerRunCommand( &cmd, MoveHelperServer() );
	pBot->SetLastUserCommand( cmd );
	pBot->pl.fixangle = FIXANGLE_NONE;

	// mirror sniper_duel_env.py's np.clip(new_pos, POSITION_LOW, POSITION_HIGH)
	// -- keep the bot inside the box the policy was actually trained on.
	Vector vecPos = pBot->GetAbsOrigin();
	Vector vecClamped = vecPos;
	vecClamped.x = clamp( vecPos.x, POSITION_LOW_X, POSITION_HIGH_X );
	vecClamped.y = clamp( vecPos.y, POSITION_LOW_Y, POSITION_HIGH_Y );
	if ( vecClamped != vecPos )
	{
		pBot->SetAbsOrigin( vecClamped );
	}
}

static ConVar sniperbot_debug( "sniperbot_debug", "0", FCVAR_CHEAT,
	"Print the sniper duel bot's observation/action vector to console periodically." );

static void DebugPrintTick( CTFPlayer *pBot, CTFPlayer *pOpponent, const float obs[SniperPolicy::kObsSize], const float action[SniperPolicy::kActionSize] )
{
	if ( !sniperbot_debug.GetBool() )
		return;

	// throttle to ~2x/sec instead of every tick
	static float s_flNextPrint = 0.0f;
	if ( gpGlobals->curtime < s_flNextPrint )
		return;
	s_flNextPrint = gpGlobals->curtime + 0.5f;

	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );
	const Vector &vecOpponentReal = pOpponent->GetAbsOrigin();

	Msg( "[sniperbot] %s raw_pos=(%.1f,%.1f) raw_yaw=%.1f rifle_active=%d opp_real_pos=(%.1f,%.1f) obs=[opp_pos=(%.1f,%.1f) opp_vis=%.0f scope_act=%.0f scope_chg=%.2f self_ang=%.1f self_hp=%.2f self_pos=(%.1f,%.1f) t_left=%.2f] action=[strafe=%.2f fwd=%.2f turn=%.2f scope=%.2f fire=%.2f]\n",
		pBot->GetPlayerName(),
		pBot->GetAbsOrigin().x, pBot->GetAbsOrigin().y,
		AngleNormalize( pBot->EyeAngles().y ),
		pRifle ? 1 : 0,
		vecOpponentReal.x, vecOpponentReal.y,
		obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9],
		action[0], action[1], action[2], action[3], action[4] );
}

static bool isRLBot( CTFPlayer *pPlayer )
{
	return pPlayer && ( pPlayer->GetFlags() & FL_FAKECLIENT ) && pPlayer->GetPlayerType() == CTFPlayer::RL_BOT;
}

// First live, non-RL-bot player found on iTeam -- this is the RL bot's
// opponent whenever a real player is on BLU, so the policy gets a live
// human's real position fed into it every tick.
static CTFPlayer *FindHumanOpponent( int iTeam )
{
	for ( int i = 1; i <= gpGlobals->maxClients; ++i )
	{
		CTFPlayer *pPlayer = ToTFPlayer( UTIL_PlayerByIndex( i ) );
		if ( !pPlayer || isRLBot( pPlayer ) || !pPlayer->IsAlive() )
			continue;

		if ( pPlayer->GetTeamNumber() != iTeam )
			continue;

		return pPlayer;
	}

	return NULL;
}

void SniperBot_RunAll()
{
	CTFPlayer *pBot = g_SniperBot.hBot.Get();
	if ( !isRLBot( pBot ) )
		return;

	if ( !pBot->IsAlive() )
	{
		g_SniperBot.bWasAlive = false;
		return;
	}

	CTFPlayer *pOpponent = FindHumanOpponent( TF_TEAM_BLUE );
	if ( !pOpponent )
		return; // nobody on BLU yet -- wait for a human to join/respawn

	MDLCACHE_CRITICAL_SECTION();

	if ( !g_SniperBot.bWasAlive )
	{
		// just respawned -- restart this bot's time_left clock
		g_SniperBot.flAliveSince = gpGlobals->curtime;
		g_SniperBot.bWasAlive = true;
	}

	float obs[SniperPolicy::kObsSize];
	BuildObservation( pBot, pOpponent, g_SniperBot.flAliveSince, obs );

	float action[SniperPolicy::kActionSize];
	SniperPolicy::Forward( obs, action );

	DebugPrintTick( pBot, pOpponent, obs, action );

	ApplyAction( pBot, pOpponent, action );
}

CON_COMMAND_F( bot_rl_solo, "Spawn (or restart) the trained-policy sniper bot on RED. Join BLU as a human to fight it.", FCVAR_CHEAT )
{
	if ( !UTIL_IsCommandIssuedByServerAdmin() )
		return;

	SniperBot_SpawnSolo();
}

CON_COMMAND_F( bot_rl_stop, "Remove the trained-policy sniper bot.", FCVAR_CHEAT )
{
	if ( !UTIL_IsCommandIssuedByServerAdmin() )
		return;

	SniperBot_RemoveDuel();
}
