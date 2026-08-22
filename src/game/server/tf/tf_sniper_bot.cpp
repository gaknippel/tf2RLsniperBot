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
// scope/fire for whichever of the two bots are currently alive. It does not
// touch team score, round counting, or respawn timing.

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

static SniperBotSlot_t g_SniperBotSlots[2]; // [0] = RED, [1] = BLU

// TF2's own spawn-point selection (run by ForceRespawn()) is what
// round_manager.nut's ResetRound() explicitly works around every round --
// it doesn't trust the engine to land a player back on "spawn_red"/
// "spawn_blu" and instead looks the named entity up and teleports there
// directly. Do the same thing here for the same reason: whatever the
// engine's spawn-point resolution is actually doing (only one candidate
// point exists per team, per the .vmf, so this shouldn't be ambiguous, but
// evidently something about it isn't reliable for a freshly-created fake
// client), pin the position ourselves instead of trusting it.
static void PinToNamedSpawn( CTFPlayer *pBot, int iTeam )
{
	const char *pszSpawnName = ( iTeam == TF_TEAM_RED ) ? "spawn_red" : "spawn_blu";
	CBaseEntity *pSpawn = gEntList.FindEntityByName( NULL, pszSpawnName );
	if ( pSpawn )
	{
		pBot->Teleport( &pSpawn->GetAbsOrigin(), &pSpawn->GetAbsAngles(), NULL );
	}
}

static CTFPlayer *SpawnOneSniperBot( int iTeam, const char *pszName )
{
	CBasePlayer *pPlayer = BotPutInServer( false, false, iTeam, TF_CLASS_SNIPER, pszName );
	if ( !pPlayer )
		return NULL;

	CTFPlayer *pBot = ToTFPlayer( pPlayer );
	pBot->SetPlayerType( CTFPlayer::RL_BOT );

	const char *pszTeamName = ( iTeam == TF_TEAM_RED ) ? "red" : "blue";
	pBot->HandleCommand_JoinTeam( pszTeamName );
	pBot->HandleCommand_JoinClass( GetPlayerClassData( TF_CLASS_SNIPER )->m_szClassName );
	pBot->ForceRespawn();
	PinToNamedSpawn( pBot, iTeam );

	return pBot;
}

void SniperBot_SpawnDuel()
{
	if ( g_SniperBotSlots[0].hBot.Get() )
	{
		g_SniperBotSlots[0].hBot->ForceRespawn();
		PinToNamedSpawn( g_SniperBotSlots[0].hBot, TF_TEAM_RED );
	}
	else
	{
		g_SniperBotSlots[0].hBot = SpawnOneSniperBot( TF_TEAM_RED, "jerry" );
	}
	g_SniperBotSlots[0].flAliveSince = gpGlobals->curtime;
	g_SniperBotSlots[0].bWasAlive = true;

	if ( g_SniperBotSlots[1].hBot.Get() )
	{
		g_SniperBotSlots[1].hBot->ForceRespawn();
		PinToNamedSpawn( g_SniperBotSlots[1].hBot, TF_TEAM_BLUE );
	}
	else
	{
		g_SniperBotSlots[1].hBot = SpawnOneSniperBot( TF_TEAM_BLUE, "terry" );
	}
	g_SniperBotSlots[1].flAliveSince = gpGlobals->curtime;
	g_SniperBotSlots[1].bWasAlive = true;
}

void SniperBot_RemoveDuel()
{
	for ( int i = 0; i < 2; ++i )
	{
		CTFPlayer *pBot = g_SniperBotSlots[i].hBot.Get();
		if ( pBot )
		{
			engine->ServerCommand( UTIL_VarArgs( "kickid %d\n", pBot->GetUserID() ) );
		}
		g_SniperBotSlots[i].hBot = NULL;
	}
}

// Builds the observation vector in the exact order tf_sniper_policy_weights.h
// documents (== export_policy.py's OBS_KEY_ORDER), from this bot's point of
// view with pOpponent as "the opponent".
//
// bMirrored must match sniper_duel_env.py's per-agent mirroring: the policy
// was retrained to always perceive itself in one canonical frame (as if
// spawned at SELF_SPAWN facing +x) regardless of which physical spawn it's
// actually on -- RED (spawn_red, facing 0 deg) matches that frame directly,
// but BLU (spawn_blu, facing 180 deg) must have its x-coordinates and yaw
// reflected before being fed to the network, or the retrained weights
// reproduce the exact same one-side-only-competent bug this was meant to
// fix. See ApplyAction for the matching un-mirror on the way back out.
static void BuildObservation( CTFPlayer *pBot, CTFPlayer *pOpponent, float flAliveSince, bool bMirrored, float obs[SniperPolicy::kObsSize] )
{
	bool bOpponentVisible = pBot->FVisible( pOpponent );

	const Vector &vecSelf = pBot->GetAbsOrigin();
	const Vector &vecOpponent = pOpponent->GetAbsOrigin();

	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	float flYaw = AngleNormalize( pBot->EyeAngles().y );
	if ( bMirrored )
	{
		flYaw = AngleNormalize( 180.0f - flYaw );
	}

	float flSelfX = bMirrored ? -vecSelf.x : vecSelf.x;
	float flOpponentX = bMirrored ? -vecOpponent.x : vecOpponent.x;

	float flTimeLeft = 1.0f - ( ( gpGlobals->curtime - flAliveSince ) / EPISODE_DURATION_SECONDS );
	flTimeLeft = clamp( flTimeLeft, 0.0f, 1.0f );

	int i = 0;
	obs[i++] = bOpponentVisible ? flOpponentX : 0.0f;
	obs[i++] = bOpponentVisible ? vecOpponent.y : 0.0f;
	obs[i++] = bOpponentVisible ? 1.0f : 0.0f;
	obs[i++] = ( pRifle && pRifle->IsZoomed() ) ? 1.0f : 0.0f;
	obs[i++] = pRifle ? pRifle->GetProgress() : 0.0f;
	obs[i++] = flYaw;
	obs[i++] = clamp( (float)pBot->GetHealth() / (float)pBot->GetMaxHealth(), 0.0f, 1.0f );
	obs[i++] = flSelfX;
	obs[i++] = vecSelf.y;
	obs[i++] = flTimeLeft;
	Assert( i == SniperPolicy::kObsSize );
}

// Turns the policy's 5-float action (strafe, forward/back, turn, scope,
// fire -- see sniper_duel_env.py's action_space comment) into a CUserCmd
// and runs it, the same way tf_bot_temp.cpp's RunPlayerMove() does for the
// waypoint bots.
//
// bMirrored must match the flag passed to BuildObservation for this same
// bot. The network's strafe/turn output is a canonical-frame intention;
// mirroring the world is a reflection, which flips chirality, so a mirrored
// bot's real-world strafe and turn must both be negated (forward/back,
// scope, and fire are unaffected -- see sniper_duel_env.py's _move_agent
// for the derivation of exactly which components flip).
static void ApplyAction( CTFPlayer *pBot, const float action[SniperPolicy::kActionSize], bool bMirrored )
{
	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	float flMirrorSign = bMirrored ? -1.0f : 1.0f;

	// Clamp frametime for the turn integration -- a single unusually long
	// server frame (e.g. a hitch right at bot spawn) would otherwise translate
	// into one giant snap-turn instead of a normal small per-tick step.
	float flTurnFrametime = MIN( gpGlobals->frametime, 0.1f );

	QAngle angViewAngles = pBot->EyeAngles();
	angViewAngles.y = AngleNormalize( angViewAngles.y + flMirrorSign * action[2] * TURN_RATE_DEG_PER_SEC * flTurnFrametime );
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

	if ( action[4] > 0.0f )
	{
		usButtons |= IN_ATTACK;
	}

	CUserCmd cmd;
	Q_memset( &cmd, 0, sizeof( cmd ) );
	VectorCopy( angViewAngles, cmd.viewangles );
	cmd.forwardmove = action[1] * pBot->MaxSpeed();
	cmd.sidemove = flMirrorSign * action[0] * pBot->MaxSpeed();
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
	"Print each sniper duel bot's observation/action vector to console periodically." );

static void DebugPrintTick( CTFPlayer *pBot, CTFPlayer *pOpponent, bool bMirrored, const float obs[SniperPolicy::kObsSize], const float action[SniperPolicy::kActionSize] )
{
	if ( !sniperbot_debug.GetBool() )
		return;

	// throttle to ~2x/sec per bot instead of every tick
	static float s_flNextPrint[2] = { 0.0f, 0.0f };
	int iSlot = ( pBot->GetTeamNumber() == TF_TEAM_RED ) ? 0 : 1;
	if ( gpGlobals->curtime < s_flNextPrint[iSlot] )
		return;
	s_flNextPrint[iSlot] = gpGlobals->curtime + 0.5f;

	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );
	const Vector &vecOpponentReal = pOpponent->GetAbsOrigin();

	Msg( "[sniperbot] %s mirrored=%d raw_pos=(%.1f,%.1f) raw_yaw=%.1f rifle_active=%d opp_real_pos=(%.1f,%.1f) obs=[opp_pos=(%.1f,%.1f) opp_vis=%.0f scope_act=%.0f scope_chg=%.2f self_ang=%.1f self_hp=%.2f self_pos=(%.1f,%.1f) t_left=%.2f] action=[strafe=%.2f fwd=%.2f turn=%.2f scope=%.2f fire=%.2f]\n",
		pBot->GetPlayerName(),
		bMirrored ? 1 : 0,
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

void SniperBot_RunAll()
{
	CTFPlayer *pRed = g_SniperBotSlots[0].hBot.Get();
	CTFPlayer *pBlue = g_SniperBotSlots[1].hBot.Get();

	if ( !isRLBot( pRed ) || !isRLBot( pBlue ) )
		return;

	MDLCACHE_CRITICAL_SECTION();

	CTFPlayer *pPair[2] = { pRed, pBlue };
	CTFPlayer *pOpponentOf[2] = { pBlue, pRed };
	// RED (spawn_red, facing 0 deg) matches the toy env's canonical SELF_SPAWN
	// frame directly; BLU (spawn_blu, facing 180 deg) needs the mirror -- see
	// BuildObservation/ApplyAction.
	bool bMirroredOf[2] = { false, true };

	for ( int i = 0; i < 2; ++i )
	{
		CTFPlayer *pBot = pPair[i];

		if ( !pBot->IsAlive() )
		{
			g_SniperBotSlots[i].bWasAlive = false;
			continue;
		}

		if ( !g_SniperBotSlots[i].bWasAlive )
		{
			// just respawned -- restart this bot's time_left clock
			g_SniperBotSlots[i].flAliveSince = gpGlobals->curtime;
			g_SniperBotSlots[i].bWasAlive = true;
		}

		float obs[SniperPolicy::kObsSize];
		BuildObservation( pBot, pOpponentOf[i], g_SniperBotSlots[i].flAliveSince, bMirroredOf[i], obs );

		float action[SniperPolicy::kActionSize];
		SniperPolicy::Forward( obs, action );

		DebugPrintTick( pBot, pOpponentOf[i], bMirroredOf[i], obs, action );

		ApplyAction( pBot, action, bMirroredOf[i] );
	}
}

CON_COMMAND_F( bot_rl_duel, "Spawn (or restart) the two trained-policy sniper-duel bots.", FCVAR_CHEAT )
{
	if ( !UTIL_IsCommandIssuedByServerAdmin() )
		return;

	SniperBot_SpawnDuel();
}

CON_COMMAND_F( bot_rl_stop, "Remove the trained-policy sniper-duel bots.", FCVAR_CHEAT )
{
	if ( !UTIL_IsCommandIssuedByServerAdmin() )
		return;

	SniperBot_RemoveDuel();
}
