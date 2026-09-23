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
	// nVariant picks which baked-in policy to run (see kVariantCount /
	// kVariantNames in the generated weights header). Out-of-range values are
	// clamped. This selects BEHAVIOUR, not difficulty -- an early checkpoint and
	// a late one duel about equally well; what visibly differs is habits like
	// scope discipline. Difficulty lives in the aim, see sniperbot_aim_skill.
	void Forward( const float obs[kObsSize], float action[kActionSize], bool bStochastic = true, int nVariant = 0 );
}

#endif // TF_SNIPER_POLICY_H
