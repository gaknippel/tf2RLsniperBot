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

struct SniperBotSlot_t
{
	CHandle<CTFPlayer> hBot;
	float flAliveSince;   // gpGlobals->curtime this life started, for the time_left approximation
	bool bWasAlive;
};

static SniperBotSlot_t g_SniperBotSlots[2]; // [0] = RED, [1] = BLU

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

	return pBot;
}

void SniperBot_SpawnDuel()
{
	if ( g_SniperBotSlots[0].hBot.Get() )
	{
		g_SniperBotSlots[0].hBot->ForceRespawn();
	}
	else
	{
		g_SniperBotSlots[0].hBot = SpawnOneSniperBot( TF_TEAM_RED, "SniperBot_RL_Red" );
	}
	g_SniperBotSlots[0].flAliveSince = gpGlobals->curtime;
	g_SniperBotSlots[0].bWasAlive = true;

	if ( g_SniperBotSlots[1].hBot.Get() )
	{
		g_SniperBotSlots[1].hBot->ForceRespawn();
	}
	else
	{
		g_SniperBotSlots[1].hBot = SpawnOneSniperBot( TF_TEAM_BLUE, "SniperBot_RL_Blue" );
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
static void ApplyAction( CTFPlayer *pBot, const float action[SniperPolicy::kActionSize] )
{
	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	QAngle angViewAngles = pBot->EyeAngles();
	angViewAngles.y = AngleNormalize( angViewAngles.y + action[2] * TURN_RATE_DEG_PER_SEC * gpGlobals->frametime );
	angViewAngles.x = 0.0f;
	angViewAngles.z = 0.0f;

	unsigned short usButtons = 0;

	// Zoom is a toggle in TF2 (single IN_ATTACK2 press flips it), not a
	// hold -- so only pulse the button on the tick our desired scope state
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
		BuildObservation( pBot, pOpponentOf[i], g_SniperBotSlots[i].flAliveSince, obs );

		float action[SniperPolicy::kActionSize];
		SniperPolicy::Forward( obs, action );

		ApplyAction( pBot, action );
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
