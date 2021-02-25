/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include "isaac_types.hpp"

namespace isaac
{

struct DefaultController
{
	static const int pass_count = 1;
	inline bool updateProjection( std::vector<isaac_dmat4>& projections, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if (first)
			projections[0] = glm::perspective( (isaac_double)45.0, (isaac_double)framebuffer_size.x/(isaac_double)framebuffer_size.y,(isaac_double)ISAAC_Z_NEAR, (isaac_double)ISAAC_Z_FAR );
		return false;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false ) {}
};

struct OrthoController
{
	static const int pass_count = 1;
	inline bool updateProjection( std::vector<isaac_dmat4>& projections, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if (first)
			setOrthographic( projections[0], (isaac_double)framebuffer_size.x/(isaac_double)framebuffer_size.y*isaac_double(2), isaac_double(2), (isaac_double)ISAAC_Z_NEAR, (isaac_double)ISAAC_Z_FAR);
		return false;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false ) {}
};

struct StereoController
{
	static const int pass_count = 2;
	StereoController() :
		eye_distance(0.06f),
		send_stereo(true)
	{}
	inline bool updateProjection( std::vector<isaac_dmat4>& projections, const isaac_size2 & framebuffer_size, json_t * const message, const bool first = false)
	{
		if ( json_t* js = json_object_get(message, "eye distance") )
		{
			send_stereo = true;
			eye_distance = json_number_value( js );
			json_object_del( message, "eye distance" );
		}
		if (first || send_stereo)
		{
			spSetPerspectiveStereoscopic( glm::value_ptr( projections[0] ), 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR, 5.0f,  eye_distance);
			spSetPerspectiveStereoscopic( glm::value_ptr( projections[1] ), 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR, 5.0f, -eye_distance);
		}
		return send_stereo;
	}
	inline void sendFeedback( json_t * const json_root, bool force = false )
	{
		if (send_stereo || force)
		{
			json_object_set_new( json_root, "eye distance", json_real( eye_distance ) );
			send_stereo = false;
		}
	}
	float eye_distance;
	bool send_stereo;
};

} //namespace isaac;
