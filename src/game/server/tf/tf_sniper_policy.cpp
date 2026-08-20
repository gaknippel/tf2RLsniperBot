#include "cbase.h"
#include "tf_sniper_policy.h"
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
}

namespace SniperPolicy
{
	void Forward( const float obs[kObsSize], float action[kActionSize] )
	{
		float hidden0[kLayer0OutputSize];
		LinearLayer<kLayer0OutputSize, kLayer0InputSize>( obs, kLayer0Weight, kLayer0Bias, kLayer0Tanh, hidden0 );

		float hidden1[kLayer1OutputSize];
		LinearLayer<kLayer1OutputSize, kLayer1InputSize>( hidden0, kLayer1Weight, kLayer1Bias, kLayer1Tanh, hidden1 );

		float rawAction[kLayer2OutputSize];
		LinearLayer<kLayer2OutputSize, kLayer2InputSize>( hidden1, kLayer2Weight, kLayer2Bias, kLayer2Tanh, rawAction );

		for ( int i = 0; i < kActionSize; ++i )
		{
			action[i] = clamp( rawAction[i], kActionLow[i], kActionHigh[i] );
		}
	}
}
