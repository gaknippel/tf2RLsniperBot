// simple 1v1 duel round controller for aiMod / mod_tf
//
// attach this to a logic_script entity in hammer:

// round stuff
MAX_ROUNDS <- 10
roundNumber <- 0
matchOver <- false
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
function ResetRound()
{
    if (matchOver)
        return

    local player = null
    while ((player = Entities.FindByClassname(player, "player")) != null)
    {
        if (!player.IsValid() || !player.IsPlayer())
            continue

        local team = player.GetTeam()
        if (team != 2 && team != 3)
            continue // skip spectators/unassigned

        // force respawn
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

    local msg = "=== Round " + roundNumber + " reset. RED: " + redWins + " BLU: " + bluWins + " ==="
    printl(msg)
    Say(null, msg, false)
}

// on death
function OnGameEvent_player_death(params)
{
    if (matchOver)
        return

    local victim = GetPlayerFromUserID(params.userid)
    local attacker = GetPlayerFromUserID(params.attacker)

    if (attacker != null && victim != null && attacker != victim)
    {
        if (attacker.GetTeam() == 2)
            redWins += 1
        else if (attacker.GetTeam() == 3)
            bluWins += 1
    }

    roundNumber += 1

    if (roundNumber >= MAX_ROUNDS)
    {
        EndMatch()
        return
    }

    //create a little delay
    local reset = function() { ResetRound() }.bindenv(this)
    CreateScheduleEvent(1.0, reset)
}

// match end
function EndMatch()
{
    matchOver = true
    local winner = "TIE"
    if (redWins > bluWins) winner = "RED"
    else if (bluWins > redWins) winner = "BLU"

    local msg = "=== match over! === RED: " + redWins + "  BLU: " + bluWins + "  WINNER: " + winner
    Say(null, msg, false)
}


// GetPlayerFromUserID(userid) is a native global function already
// exposed to script (see Script_GetPlayerFromUserID in vscript_server.cpp) -
// do not redefine it here


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


    //force custom cfg for 1v1 map
    SendToConsole("exec 1v1map")

    //creates the ai rl bot
    CreateScheduleEvent(1.0, function() { SendToConsole("tf_bot_add red sniper") })

    printl("=== round_manager.nut loaded. max rounds: " + MAX_ROUNDS + " ===")
}

Init()
