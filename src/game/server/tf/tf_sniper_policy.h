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
	//
	// bStochastic selects which policy you get, and the two are NOT close:
	//
	//   true  (default) -- sample from N(mean, kActionStd), exactly what PPO
	//                      optimised and exactly what the training reward
	//                      curves describe. USE THIS.
	//   false           -- the bare network mean. Useful for debugging a single
	//                      tick's output, not for playing: the fire dimension's
	//                      mean never crosses its 0.0 threshold, so a bot run
	//                      this way holds a scoped, charged, on-target shot and
	//                      never fires. Measured win rate across difficulties
	//                      0.00/0.50/1.00 was 26.7/60.0/53.3 percent for the
	//                      mean versus 96.7/93.3/66.7 for samples.
	void Forward( const float obs[kObsSize], float action[kActionSize], bool bStochastic = true );
}

#endif // TF_SNIPER_POLICY_H
