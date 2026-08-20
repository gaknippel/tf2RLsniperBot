//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Spawns and drives the two trained-policy sniper-duel bots. This
// is a separate per-tick path from Bot_RunAll()'s waypoint-following bot AI
// (tf_bot_temp.cpp) -- these bots are tagged CTFPlayer::RL_BOT specifically
// so Bot_RunAll() skips them, and SniperBot_RunAll() drives their usercmd
// directly from tf_sniper_policy.h's forward pass instead.
//
//=============================================================================

#ifndef TF_SNIPER_BOT_H
#define TF_SNIPER_BOT_H
#ifdef _WIN32
#pragma once
#endif

// Spawns (or respawns, if already present) the RED/BLU sniper-duel pair.
void SniperBot_SpawnDuel();

// Removes both duel bots, if present.
void SniperBot_RemoveDuel();

// Call once per server frame (alongside Bot_RunAll()) to drive both bots.
void SniperBot_RunAll();

#endif // TF_SNIPER_BOT_H
