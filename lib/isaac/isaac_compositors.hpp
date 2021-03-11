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

namespace isaac
{

    struct DefaultCompositor
    {
        DefaultCompositor( isaac_size2 framebufferSize ) {}
        static inline isaac_size2 getCompositedbufferSize( isaac_size2 framebufferSize )
        {
            return framebufferSize;
        }
        inline uint32_t* doCompositing(IceTImage* image)
        {
            return icetImageGetColorui(image[0]);
        }
    };

    template <typename T_Controller>
    class StereoCompositorSideBySide
    {
        public:
            static inline isaac_size2 getCompositedbufferSize( isaac_size2 framebufferSize )
            {
                isaac_size2 compbufferSize =
                {
                    framebufferSize.x * 2,
                    framebufferSize.y
                };
                return compbufferSize;
            }
            StereoCompositorSideBySide( isaac_size2 framebufferSize ) :
                framebufferSize( framebufferSize ),
                compbufferSize( getCompositedbufferSize( framebufferSize ) )
            {
                compbuffer = (uint32_t*)malloc(sizeof(uint32_t) * compbufferSize.x * compbufferSize.y);
            }
            ~StereoCompositorSideBySide()
            {
                free(compbuffer);
            }
            inline uint32_t* doCompositing(IceTImage* image)
            {
                static_assert(T_Controller::passCount >= 2, "Not enough passes defined in Controller for StereoCompositor!");
                uint32_t* left = icetImageGetColorui(image[0]);
                uint32_t* right = icetImageGetColorui(image[1]);
                for (unsigned int y = 0; y < compbufferSize.y; y++)
                {
                    memcpy( &(compbuffer[y*compbufferSize.x                    ]), &( left[y*framebufferSize.x]), sizeof(uint32_t) * framebufferSize.x);
                    memcpy( &(compbuffer[y*compbufferSize.x + framebufferSize.x]), &(right[y*framebufferSize.x]), sizeof(uint32_t) * framebufferSize.x);
                }
                return compbuffer;
            }
        private:
            isaac_size2 compbufferSize;
            isaac_size2 framebufferSize;
            uint32_t* compbuffer;
    };

    template <
        typename T_Controller,
        uint32_t LeftFilter,
        uint32_t RightFilter
    >
    class StereoCompositorAnaglyph
    {
        public:
            static inline isaac_size2 getCompositedbufferSize( isaac_size2 framebufferSize )
            {
                return framebufferSize;
            }
            StereoCompositorAnaglyph( isaac_size2 framebufferSize ) :
                framebufferSize( framebufferSize )
            {
                compbuffer = (uint32_t*)malloc(sizeof(uint32_t) * framebufferSize.x * framebufferSize.y);
            }
            ~StereoCompositorAnaglyph()
            {
                free(compbuffer);
            }
            inline uint32_t* doCompositing(IceTImage* image)
            {
                static_assert(T_Controller::passCount >= 2, "Not enough passes defined in Controller for StereoCompositor!");
                uint32_t* left = icetImageGetColorui(image[0]);
                uint32_t* right = icetImageGetColorui(image[1]);
                for (unsigned int x = 0; x < framebufferSize.x; x++)
                    for (unsigned int y = 0; y < framebufferSize.y; y++)
                        compbuffer[x+y*framebufferSize.x] =
                            ( left[x+y*framebufferSize.x] &  LeftFilter) |
                            (right[x+y*framebufferSize.x] & RightFilter);
                return compbuffer;
            }
        private:
            isaac_size2 framebufferSize;
            uint32_t* compbuffer;
    };

} //namespace isaac;
