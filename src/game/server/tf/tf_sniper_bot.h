//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Spawns and drives the trained-policy sniper bot. This is a
// separate per-tick path from Bot_RunAll()'s waypoint-following bot AI
// (tf_bot_temp.cpp) -- the bot is tagged CTFPlayer::RL_BOT specifically
// so Bot_RunAll() skips it, and SniperBot_RunAll() drives its usercmd
// directly from tf_sniper_policy.h's forward pass instead.
//
// There is exactly one trained role (RED); a human on BLU is the opponent.
// See tf_sniper_bot.cpp's file-header comment for why self-play/mirroring
// was dropped.
//
//=============================================================================

#ifndef TF_SNIPER_BOT_H
#define TF_SNIPER_BOT_H
#ifdef _WIN32
#pragma once
#endif

// Spawns (or respawns, if already present) the RL bot on RED. Join BLU as a
// human to fight it -- SniperBot_RunAll() picks up whichever live human is
// on BLU as its opponent.
void SniperBot_SpawnSolo();

// Removes the RL bot, if present.
void SniperBot_RemoveDuel();

// Call once per server frame (alongside Bot_RunAll()) to drive the bot.
void SniperBot_RunAll();

#endif // TF_SNIPER_BOT_H
