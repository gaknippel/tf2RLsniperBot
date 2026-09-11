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

// 2026-09-10: aiming is computed here analytically instead of being taken from
// the policy's turn output, and firing is gated on actually being on target.
//
// Aiming is not a learning problem -- it is one line of trigonometry, exact at
// every position on the map. Training a network to approximate atan2 from
// sampled states went badly across two full 50M-step runs: the first learned no
// aiming at all (it always spawned facing the opponent, so "drift one way and
// fire when the target crosses the crosshair" scored a 98% in-sim hit rate
// while sitting 73 degrees off target in game), and even a policy cloned
// directly from a perfect analytic expert only turned the right way in ~65% of
// swept aim errors, because a 2D toy env cannot cover the state space the real
// game presents.
//
// So the split is now: the bridge owns aim (exact, from live engine geometry,
// with tunable imperfection below so it stays beatable), and the policy owns
// the decisions that genuinely need learning and have no closed form --
// positioning, when to push or hold, when to break line of sight, when to take
// the shot. That also removes the sim-to-real aiming gap permanently, since aim
// no longer depends on anything the toy env approximated.
//
// AIM_* knobs exist so this is a fightable opponent rather than a literal
// aimbot, and they are far easier to tune for feel than hoping PPO lands
// somewhere fun:
//   SLEW            - max degrees/sec the view can swing, so it cannot teleport
//                     onto a target; lower feels more human.
//   ERROR_UNITS     - how far off the aim point it settles, measured in world
//                     units AT THE TARGET, re-rolled per target acquisition.
//   SETTLE_UNITS    - how close counts as "on target" for the fire gate, also
//                     in world units at the target.
//   MIN_CHARGE      - scope charge required before taking a shot. NOT about
//                     damage: an UNCHARGED headshot already does
//                     TF_WEAPON_SNIPERRIFLE_DAMAGE_MIN (50) x3 = 150, which
//                     outright kills a 125 HP sniper, so waiting for a full
//                     charge buys nothing and just makes the bot passive. This
//                     exists only to clear the engine's headshot rules in
//                     CTFSniperRifle::CanFireCriticalShot(): crits need the
//                     scope up AND at least
//                     TF_WEAPON_SNIPERRIFLE_NO_CRIT_AFTER_ZOOM_TIME (0.2s)
//                     since the zoom began. Charge rises at
//                     CHARGE_PER_SEC (50) toward DAMAGE_MAX (150), so charge
//                     fraction == seconds_zoomed / 3 -- meaning 0.2s is only
//                     0.067. 0.15 is ~0.45s, a comfortable margin past the
//                     window while still shooting ~6x sooner than a full
//                     charge would. (Full charge is only actually required for
//                     weapons carrying the sniper_no_headshot_without_full_charge
//                     attribute, i.e. the Machina -- not a stock rifle.)
//
// 2026-09-10: ERROR and SETTLE are deliberately expressed in world units
// rather than degrees. They were degrees first (2.0 and 3.5), which looks
// reasonable but is distance-blind: a player is only ~49 units wide, so at 1000
// units their half-width subtends about 1.4 degrees. A fixed 2-degree offset is
// therefore a guaranteed miss at range while being harmless up close, which is
// exactly what live testing showed -- the bot tracked well and still landed
// "pixels off" at distance. Converting a linear tolerance into an angle per
// tick (atan(units / distance)) makes it tight at long range and forgiving
// close up, which is both more accurate and more human.
static const float AIM_SLEW_DEG_PER_SEC = 220.0f;
static const float AIM_ERROR_UNITS = 6.0f;
static const float AIM_SETTLE_UNITS = 14.0f;
static const float AIM_MIN_CHARGE_TO_FIRE = 0.15f;

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
	// see the AIM_* constants -- a per-acquisition aim offset, re-rolled each
	// time the bot reacquires a target, so it doesn't settle pixel-perfect.
	float flAimErrorOffsetUnits;
	bool bHadTargetLastTick;
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
	// fresh life, fresh aim state -- see the AIM_* constants
	g_SniperBot.bHadTargetLastTick = false;
	g_SniperBot.flAimErrorOffsetUnits = 0.0f;
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

	// 2026-09-10: egocentric aim error, added after the previous policy turned
	// out never to have learned to aim -- see sniper_duel_env.py's
	// aim_error_sin/cos comment. Signed error between where the bot faces and
	// the true bearing to its opponent, so positive always means "turn
	// positive to correct" regardless of where either of them is standing.
	// Zeroed without visibility, matching opponent_pos, so the bot gets no aim
	// cue through a wall. Must stay numerically identical to the Python
	// _get_obs computation or the policy is reading a different quantity than
	// it trained on.
	float flAimErrorSin = 0.0f;
	float flAimErrorCos = 0.0f;
	if ( bOpponentVisible )
	{
		float flBearing = RAD2DEG( atan2f( vecOpponent.y - vecSelf.y, vecOpponent.x - vecSelf.x ) );
		float flErr = AngleNormalize( flBearing - flYaw );
		flAimErrorSin = sinf( DEG2RAD( flErr ) );
		flAimErrorCos = cosf( DEG2RAD( flErr ) );
	}

	// NOTE: this order is gymnasium.spaces.Dict's alphabetical key order, NOT
	// the declaration order in SniperDuelEnv -- it must match export_policy.py's
	// OBS_KEY_ORDER exactly. The aim_error_* keys sort to the FRONT, which
	// shifts every field that follows them.
	int i = 0;
	obs[i++] = flAimErrorCos;
	obs[i++] = flAimErrorSin;
	obs[i++] = bOpponentVisible ? vecOpponent.x : 0.0f;
	obs[i++] = bOpponentVisible ? vecOpponent.y : 0.0f;
	obs[i++] = bOpponentVisible ? 1.0f : 0.0f;
	obs[i++] = ( pRifle && pRifle->IsZoomed() ) ? 1.0f : 0.0f;
	// 2026-09-10: was pRifle->GetProgress(), which is NOT the scope charge --
	// CTFSniperRifle::GetProgress() returns GetRageMeter()/100, the Hitman's
	// Heatmaker rage meter, which is always 0 on a stock rifle. So this
	// observation fed the policy a constant 0 for its entire training-to-
	// deployment life, and live debug showed scope_charge pinned at 0.00 every
	// tick while the bot was demonstrably zoomed. GetScopeChargePerc() is the
	// real thing: m_flChargedDamage / TF_WEAPON_SNIPERRIFLE_DAMAGE_MAX, i.e.
	// 0..1. It was added to tf_weapon_sniperrifle.h for this -- the equivalent
	// GetHUDDamagePerc() exists but is CLIENT_DLL only.
	obs[i++] = pRifle ? pRifle->GetScopeChargePerc() : 0.0f;
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
// 2026-09-10: the policy's turn output (action[2]) is deliberately IGNORED --
// see the AIM_* constants at the top of this file for why aiming moved here.
// The policy still drives movement, scope and the intent to fire.
static void ApplyAction( CTFPlayer *pBot, CTFPlayer *pOpponent, const float action[SniperPolicy::kActionSize] )
{
	CTFSniperRifle *pRifle = dynamic_cast< CTFSniperRifle * >( pBot->GetActiveTFWeapon() );

	// Clamp frametime for the turn integration -- a single unusually long
	// server frame (e.g. a hitch right at bot spawn) would otherwise translate
	// into one giant snap-turn instead of a normal small per-tick step.
	float flTurnFrametime = MIN( gpGlobals->frametime, 0.1f );

	bool bOpponentVisible = pBot->FVisible( pOpponent );

	// Re-roll the steady-state aim offset each time a target is reacquired, so
	// the bot doesn't converge on the exact same offset every fight. Stored in
	// world units at the target and converted to an angle per tick below, so it
	// stays correct at any range -- see the AIM_* comment.
	if ( bOpponentVisible && !g_SniperBot.bHadTargetLastTick )
	{
		g_SniperBot.flAimErrorOffsetUnits = RandomFloat( -AIM_ERROR_UNITS, AIM_ERROR_UNITS );
	}
	g_SniperBot.bHadTargetLastTick = bOpponentVisible;

	QAngle angViewAngles = pBot->EyeAngles();

	// Analytic aim: slew the view toward the opponent's actual position at a
	// bounded rate. Pitch is driven too -- the policy's 2D world had no concept
	// of it, so a learned turn could never have aimed up or down at all.
	float flAimErrorDeg = 180.0f;
	float flSettleToleranceDeg = 0.0f;
	if ( bOpponentVisible )
	{
		Vector vecAimAt = pOpponent->EyePosition();
		Vector vecToTarget = vecAimAt - pBot->EyePosition();
		float flRange = MAX( vecToTarget.Length(), 1.0f );

		// Convert the linear tolerances into angles for THIS range, so a fixed
		// number of world units means the same thing point-blank and across the
		// map -- see the AIM_* comment above.
		float flErrorDeg = RAD2DEG( atanf( g_SniperBot.flAimErrorOffsetUnits / flRange ) );
		flSettleToleranceDeg = RAD2DEG( atanf( AIM_SETTLE_UNITS / flRange ) );

		QAngle angWanted;
		VectorAngles( vecToTarget, angWanted );
		angWanted.y = AngleNormalize( angWanted.y + flErrorDeg );

		float flMaxStep = AIM_SLEW_DEG_PER_SEC * flTurnFrametime;
		float flYawDelta = AngleNormalize( angWanted.y - angViewAngles.y );
		float flPitchDelta = AngleNormalize( angWanted.x - angViewAngles.x );

		angViewAngles.y = AngleNormalize( angViewAngles.y + clamp( flYawDelta, -flMaxStep, flMaxStep ) );
		angViewAngles.x = AngleNormalize( angViewAngles.x + clamp( flPitchDelta, -flMaxStep, flMaxStep ) );

		// how far off we still are AFTER this tick's slew -- the fire gate below
		// keys off this so the bot can't shoot mid-swing.
		flAimErrorDeg = MAX( fabsf( AngleNormalize( angWanted.y - angViewAngles.y ) ),
		                     fabsf( AngleNormalize( angWanted.x - angViewAngles.x ) ) );
	}
	else
	{
		// no target: let the policy's movement carry it, and keep the view level
		// so it isn't left staring at the floor when it reacquires.
		angViewAngles.x = Approach( 0.0f, angViewAngles.x, AIM_SLEW_DEG_PER_SEC * flTurnFrametime );
	}
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

	// Fire gate. The policy supplies the INTENT to shoot; the bridge decides
	// whether pulling the trigger now would actually be a shot rather than a
	// wasted one. Three conditions, all mechanical:
	//
	//  - the opponent is really visible (engine trace, not a learned habit);
	//  - the view has actually settled on them, so it can't fire mid-swing and
	//    spray walls -- which is exactly what the previous build did, holding
	//    the trigger while 32 degrees off target;
	//  - the scope is charged enough to headshot IF we're scoped at all.
	//
	// That last one matters more than it looks: the rifle zeroes its charge on
	// every shot, so a policy that holds the trigger down never accumulates any
	// charge and can only ever land body shots. Live debug showed exactly that,
	// scope_charge pinned at 0.00 on every single tick. Waiting for the charge
	// is what makes a scoped bot lethal instead of an annoyance.
	// 2026-09-10: requires being SCOPED, where this first read "!bIsZoomed ||
	// ...". That disjunction meant the instant the rifle unscoped -- which it
	// does after every shot -- the gate went true and the bot immediately fired
	// again unscoped, so it body-shot forever and never scoped at all. Live
	// debug showed scope_act flicking to 0 through whole fights. Scoping is
	// mandatory for a headshot crit per the engine, and the 3x multiplier is the
	// entire point of the class.
	//
	// The charge requirement is only the engine's 0.2s post-zoom crit lockout,
	// NOT a damage consideration -- see AIM_MIN_CHARGE_TO_FIRE. It was 0.90
	// briefly, which made the bot sit there for ~2.7s per shot for no benefit,
	// since an uncharged headshot (150) already kills a 125 HP sniper outright.
	bool bOnTarget = bOpponentVisible && flAimErrorDeg <= flSettleToleranceDeg;
	bool bReadyToShoot = bIsZoomed && pRifle && pRifle->GetScopeChargePerc() >= AIM_MIN_CHARGE_TO_FIRE;
	if ( action[4] > 0.0f && bOnTarget && bReadyToShoot )
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

	// 2026-09-10: the obs indices below MUST track BuildObservation's layout.
	// They silently didn't after aim_error_cos/sin were prepended (kObsSize 10
	// -> 12), so every field printed one or two slots out of place -- self_pos
	// showed (yaw, health), t_left showed self x, and scope_charge showed the
	// visibility flag. An hour went into chasing a "broken" observation that was
	// in fact correct; only the printout was wrong. aim_err_deg is derived back
	// out of the sin/cos pair since that's the number worth eyeballing.
	float flAimErrDeg = RAD2DEG( atan2f( obs[1], obs[0] ) );

	Msg( "[sniperbot] %s raw_pos=(%.1f,%.1f) raw_yaw=%.1f pitch=%.1f rifle=%d opp_real=(%.1f,%.1f) "
	     "obs=[aim_err=%.1f opp_pos=(%.1f,%.1f) opp_vis=%.0f scope_act=%.0f scope_chg=%.2f "
	     "self_ang=%.1f self_hp=%.2f self_pos=(%.1f,%.1f) t_left=%.2f] "
	     "action=[strafe=%.2f fwd=%.2f turn=%.2f(ignored) scope=%.2f fire=%.2f]\n",
		pBot->GetPlayerName(),
		pBot->GetAbsOrigin().x, pBot->GetAbsOrigin().y,
		AngleNormalize( pBot->EyeAngles().y ),
		AngleNormalize( pBot->EyeAngles().x ),
		pRifle ? 1 : 0,
		vecOpponentReal.x, vecOpponentReal.y,
		flAimErrDeg,
		obs[2], obs[3],          // opponent_pos
		obs[4],                  // opponent_visible
		obs[5],                  // scope_active
		obs[6],                  // scope_charge
		obs[7],                  // self_angle
		obs[8],                  // self_health
		obs[9], obs[10],         // self_pos
		obs[11],                 // time_left
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
