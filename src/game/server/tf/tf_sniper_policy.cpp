#include "cbase.h"
#include "tf_sniper_policy.h"
#include "vstdlib/random.h"
#include <cmath>

// The exported policy is always policy_net = [Linear, Tanh, Linear, Tanh]
// followed by action_net = [Linear] (see export_policy.py) -- i.e. exactly
// 3 layers. Written out explicitly rather than looping over a generic list
// of layers, since each layer's weight matrix is a differently-shaped
// fixed-size C array. If the model architecture ever changes, this static_assert
// catches it at compile time.
static_assert( SniperPolicy::kLayerCount == 3, "tf_sniper_policy.cpp assumes exactly 3 layers -- update Forward() if the exported architecture changed." );

namespace
{
	template< int OUT_DIM, int IN_DIM >
	void LinearLayer( const float in[IN_DIM], const float weight[OUT_DIM][IN_DIM], const float bias[OUT_DIM], bool applyTanh, float out[OUT_DIM] )
	{
		for ( int o = 0; o < OUT_DIM; ++o )
		{
			float sum = bias[o];
			for ( int i = 0; i < IN_DIM; ++i )
			{
				sum += weight[o][i] * in[i];
			}
			out[o] = applyTanh ? tanhf( sum ) : sum;
		}
	}

	// Standard-normal sample via Box-Muller, on top of the engine's uniform RNG
	// so nothing here needs its own seeding or state. Both generated values are
	// used rather than discarding one, since kActionSize draws happen per tick.
	void StandardNormalPair( float &z0, float &z1 )
	{
		// guard the log against exactly 0.0, which RandomFloat can return
		float u1 = RandomFloat( 1.0e-7f, 1.0f );
		float u2 = RandomFloat( 0.0f, 1.0f );
		float r = sqrtf( -2.0f * logf( u1 ) );
		float theta = 2.0f * M_PI_F * u2;
		z0 = r * cosf( theta );
		z1 = r * sinf( theta );
	}
}

namespace SniperPolicy
{
	void Forward( const float obs[kObsSize], float action[kActionSize], bool bStochastic )
	{
		float hidden0[kLayer0OutputSize];
		LinearLayer<kLayer0OutputSize, kLayer0InputSize>( obs, kLayer0Weight, kLayer0Bias, kLayer0Tanh, hidden0 );

		float hidden1[kLayer1OutputSize];
		LinearLayer<kLayer1OutputSize, kLayer1InputSize>( hidden0, kLayer1Weight, kLayer1Bias, kLayer1Tanh, hidden1 );

		float rawAction[kLayer2OutputSize];
		LinearLayer<kLayer2OutputSize, kLayer2InputSize>( hidden1, kLayer2Weight, kLayer2Bias, kLayer2Tanh, rawAction );

		// PPO's actor network outputs the MEAN of a Gaussian over actions. The
		// policy that was actually trained -- the one the reward curves and the
		// eval numbers describe -- takes a SAMPLE from that Gaussian and clips it
		// to the action bounds, which is what this reproduces.
		//
		// Running the mean instead is not a minor simplification, it is a
		// different agent. See the comment on Forward() in the header for the
		// measured gap; the short version is that the fire dimension's mean sits
		// around -0.8 and never reaches its 0.0 threshold, so the mean-only bot
		// never pulls the trigger. The trigger rate the policy learned (~12% of
		// ticks) lives entirely in kActionStd.
		if ( bStochastic )
		{
			float z[kActionSize + 1];
			for ( int i = 0; i < kActionSize; i += 2 )
			{
				StandardNormalPair( z[i], z[i + 1] );
			}
			for ( int i = 0; i < kActionSize; ++i )
			{
				rawAction[i] += kActionStd[i] * z[i];
			}
		}

		for ( int i = 0; i < kActionSize; ++i )
		{
			action[i] = clamp( rawAction[i], kActionLow[i], kActionHigh[i] );
		}
	}
}
