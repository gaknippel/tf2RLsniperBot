// simple 1v1 duel round controller for aiMod / mod_tf
//
// attach this to a logic_script entity in hammer.
//
// Endless duel: every death resets both players and the score carries on
// forever, posted to chat after each round. There is no match end -- reload the
// map to zero the score.
//
// Only VScript methods this build actually binds are used here. GetPlayerName
// is NOT exposed to script in this SDK (C++-only), and calling it threw a
// Squirrel error that killed ResetRound mid-loop and left players stuck in the
// air -- so rounds are reported by team, not by name.

roundNumber <- 0
redWins <- 0
bluWins <- 0

// spawn setup
// - name your two spawn entities "spawn_red" and "spawn_blu"
// in hammer (name field in object properties), then this script
// will look them up by targetname.
function GetSpawnPoint(teamNum)
{
    local name = (teamNum == 2) ? "spawn_red" : "spawn_blu"
    local ent = Entities.FindByName(null, name)
    return ent
}

// round reset
//
// tf_sniper_bot.cpp relies on this exact behaviour: everyone force-respawned
// and pinned to their team's spawn entity. The bot detects the respawn on the
// following tick and applies its own spawn spread after this has run, so keep
// the respawn/teleport here even though the bot overrides its own position.
function ResetRound()
{
    local player = null
    while ((player = Entities.FindByClassname(player, "player")) != null)
    {
        if (!player.IsValid() || !player.IsPlayer())
            continue

        local team = player.GetTeam()
        if (team != 2 && team != 3)
            continue // skip spectators/unassigned

        player.ForceRespawn()

        local spawn = GetSpawnPoint(team)
        if (spawn != null)
        {
            player.SetAbsOrigin(spawn.GetOrigin())
            player.SetAbsAngles(spawn.GetAbsAngles())
        }

        //full heal just like in MGE
        player.SetHealth(player.GetMaxHealth())
    }
}

function PostScore(roundWinner)
{
    local msg = "Round " + roundNumber + ": " + roundWinner + "  |  RED " + redWins + " - " + bluWins + " BLU"
    printl(msg)
    Say(null, msg, false)
}

// on death
function OnGameEvent_player_death(params)
{
    local victim = GetPlayerFromUserID(params.userid)
    local attacker = GetPlayerFromUserID(params.attacker)

    roundNumber += 1

    local roundWinner = "no point (suicide/world)"
    if (attacker != null && victim != null && attacker != victim)
    {
        if (attacker.GetTeam() == 2)
        {
            redWins += 1
            roundWinner = "RED wins"
        }
        else if (attacker.GetTeam() == 3)
        {
            bluWins += 1
            roundWinner = "BLU wins"
        }
    }

    PostScore(roundWinner)

    //create a little delay
    local reset = function() { ResetRound() }.bindenv(this)
    CreateScheduleEvent(1.0, reset)
}


scheduledEvents <- []

function CreateScheduleEvent(delay, func)
{
    scheduledEvents.append({ time = Time() + delay, fn = func })
}

function Think()
{
    local i = 0
    while (i < scheduledEvents.len())
    {
        if (Time() >= scheduledEvents[i].time)
        {
            scheduledEvents[i].fn()
            scheduledEvents.remove(i)
        }
        else
        {
            i += 1
        }
    }
    return -1 // run every server frame
}

function Init()
{
    __CollectGameEventCallbacks(this)

    // SendToServerConsole, NOT SendToConsole. SendToConsole delivers to
    // "the listen server host", which the engine takes to mean player slot 1 --
    // and SourceTV (tv_enable 1) joins first and occupies slot 1. Both commands
    // below were being sent to SourceTV's console, a fake client that ignores
    // them: 1v1map.cfg never ran and the bot never spawned. Server-console
    // commands don't depend on slot order. On a listen server they share the
    // local command buffer, so the binds in 1v1map.cfg still apply.
    //
    // Requires sv_allow_point_servercommand always (set in autoexec.cfg);
    // otherwise TF only permits this on official Valve maps and it does nothing.

    //force custom cfg for 1v1 map
    SendToServerConsole("exec 1v1map")

    //spawns the trained-policy RL sniper bot on RED (see tf_sniper_bot.cpp)
    //not the official valve bot with nextbot ai -- join BLU to fight it
    CreateScheduleEvent(1.0, function() { SendToServerConsole("bot_rl_solo") })

    printl("=== round_manager.nut loaded. endless duel, score in chat ===")
}

Init()
