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

#include "isaac/isaac_helper.hpp"
#include "isaac_types.hpp"

namespace isaac
{
    struct DefaultController
    {
        static const int passCount = 1;
        inline bool updateProjection(
            std::vector<isaac_dmat4>& projections,
            const isaac_size2& framebufferSize,
            json_t* const message,
            const bool first = false)
        {
            if(first)
                projections[0] = glm::perspective(
                    (isaac_double) 45.0,
                    (isaac_double) framebufferSize.x / (isaac_double) framebufferSize.y,
                    (isaac_double) ISAAC_Z_NEAR,
                    (isaac_double) ISAAC_Z_FAR);
            return false;
        }
        inline void sendFeedback(json_t* const jsonRoot, bool force = false)
        {
        }
    };

    struct OrthoController
    {
        static const int passCount = 1;
        inline bool updateProjection(
            std::vector<isaac_dmat4>& projections,
            const isaac_size2& framebufferSize,
            json_t* const message,
            const bool first = false)
        {
            if(first)
                setOrthographic(
                    projections[0],
                    (isaac_double) framebufferSize.x / (isaac_double) framebufferSize.y * isaac_double(2),
                    isaac_double(2),
                    (isaac_double) ISAAC_Z_NEAR,
                    (isaac_double) ISAAC_Z_FAR);
            return false;
        }
        inline void sendFeedback(json_t* const jsonRoot, bool force = false)
        {
        }
    };

    struct StereoController
    {
        static const int passCount = 2;
        StereoController() : eyeDistance(0.06f), sendStereo(true)
        {
        }
        inline bool updateProjection(
            std::vector<isaac_dmat4>& projections,
            const isaac_size2& framebufferSize,
            json_t* const message,
            const bool first = false)
        {
            if(json_t* js = json_object_get(message, "eye distance"))
            {
                sendStereo = true;
                eyeDistance = json_number_value(js);
                json_object_del(message, "eye distance");
            }
            if(first || sendStereo)
            {
                spSetPerspectiveStereoscopic(
                    glm::value_ptr(projections[0]),
                    45.0f,
                    (isaac_float) framebufferSize.x / (isaac_float) framebufferSize.y,
                    ISAAC_Z_NEAR,
                    ISAAC_Z_FAR,
                    5.0f,
                    eyeDistance);
                spSetPerspectiveStereoscopic(
                    glm::value_ptr(projections[1]),
                    45.0f,
                    (isaac_float) framebufferSize.x / (isaac_float) framebufferSize.y,
                    ISAAC_Z_NEAR,
                    ISAAC_Z_FAR,
                    5.0f,
                    -eyeDistance);
            }
            return sendStereo;
        }
        inline void sendFeedback(json_t* const jsonRoot, bool force = false)
        {
            if(sendStereo || force)
            {
                json_object_set_new(jsonRoot, "eye distance", json_real(eyeDistance));
                sendStereo = false;
            }
        }
        float eyeDistance;
        bool sendStereo;
    };

} // namespace isaac
