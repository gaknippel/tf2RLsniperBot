//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose: Forward pass for the trained sniper-duel PPO policy (see
// python/toy_env/). Pure math over the baked-in weights in
// tf_sniper_policy_weights.h -- no game-state knowledge here, that lives in
// whatever builds the observation vector and interprets the action vector.
//
//=============================================================================

#ifndef TF_SNIPER_POLICY_H
#define TF_SNIPER_POLICY_H
#ifdef _WIN32
#pragma once
#endif

#include "tf_sniper_policy_weights.h"

namespace SniperPolicy
{
	// obs must be kObsSize floats, laid out in the exact order documented at
	// the bottom of tf_sniper_policy_weights.h. action receives kActionSize
	// floats, already clamped to [kActionLow, kActionHigh].
	void Forward( const float obs[kObsSize], float action[kActionSize] );
}

#endif // TF_SNIPER_POLICY_H
