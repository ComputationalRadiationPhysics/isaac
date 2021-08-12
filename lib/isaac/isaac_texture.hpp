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

#include "isaac_helper.hpp"
#include "isaac_types.hpp"


namespace isaac
{
    enum class FilterType
    {
        NEAREST,
        LINEAR
    };

    enum class BorderType
    {
        CLAMP,
        REPEAT,
        VALUE
    };

    enum class IndexType
    {
        SWEEP,
        MORTON
    };

    /**
     * @brief Software texture implementation
     *
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_Type, int T_textureDim, IndexType T_indexType = IndexType::SWEEP>
    class Texture
    {
    public:
        Texture() = default;

        /**
         * @brief Initialize texture
         *
         * @param bufferPtr Valid pointer to free memory
         * @param size Size of the texture in T_TextureDim dimensions
         * @param guardSize Size of the memory access guard, default = 0
         */
        ISAAC_HOST_DEVICE_INLINE void init(
            T_Type* bufferPtr,
            const isaac_size_dim<T_textureDim>& size,
            const isaac_size_dim<T_textureDim>& guardSize = isaac_size_dim<T_textureDim>(0))
        {
            this->bufferPtr = bufferPtr;
            this->size = size;
            this->sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;
            this->guardSize = guardSize;
        }

        ISAAC_HOST_DEVICE_INLINE T_Type operator[](const isaac_int_dim<T_textureDim>& coord) const
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int_dim<T_textureDim>(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }


        ISAAC_HOST_DEVICE_INLINE T_Type& operator[](const isaac_int_dim<T_textureDim>& coord)
        {
            isaac_uint_dim<T_textureDim> offsetCoord = coord + isaac_int_dim<T_textureDim>(guardSize);
            assert(isInUpperBounds(offsetCoord, sizeWithGuard));
            return bufferPtr[hash(offsetCoord)];
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<1>& coord) const
        {
            return coord.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<2>& coord) const
        {
            if(T_indexType == IndexType::MORTON)
                return (part1By1(coord.y) << 1) + part1By1(coord.x);
            else
                return coord.x + coord.y * sizeWithGuard.x;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_uint hash(const isaac_uint_dim<3>& coord) const
        {
            if(T_indexType == IndexType::MORTON)
                return (part1By2(coord.z) << 2) | (part1By2(coord.y) << 1) | part1By2(coord.x);
            else
                return coord.x + coord.y * sizeWithGuard.x + coord.z * sizeWithGuard.x * sizeWithGuard.y;
        }

        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<T_textureDim> getSize() const
        {
            return size;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<T_textureDim> getSizeWithGuard() const
        {
            return sizeWithGuard;
        }
        ISAAC_HOST_DEVICE_INLINE isaac_size_dim<T_textureDim> getGuardSize() const
        {
            return guardSize;
        }

        ISAAC_HOST_DEVICE_INLINE T_Type* getPtr() const
        {
            return bufferPtr;
        }

    private:
        T_Type* bufferPtr = nullptr;
        isaac_size_dim<T_textureDim> size;
        isaac_size_dim<T_textureDim> sizeWithGuard;
        isaac_size_dim<T_textureDim> guardSize;
    };


    template<FilterType T_filter, BorderType T_border, int T_normalized = 0>
    class Sampler
    {
    public:
        template<typename T_Type, int T_textureDim, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_float sample(
            const Texture<T_Type, T_textureDim, T_indexType>& texture,
            const isaac_float_dim<T_textureDim>& coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            isaac_float result;
            if(T_filter == FilterType::LINEAR)
            {
                result = interpolate(texture, coord, borderValue);
            }
            else
            {
                result = isaac_float(safeMemoryAccess(texture, isaac_int_dim<T_textureDim>(coord), borderValue));
            }
            if(T_normalized)
                result /= isaac_float(std::numeric_limits<T_Type>::max());
            return result;
        }

        template<int T_n, typename T_CompType, int T_textureDim, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_vec_dim<T_n, isaac_float> sample(
            const Texture<isaac_vec_dim<T_n, T_CompType>, T_textureDim, T_indexType>& texture,
            const isaac_float_dim<T_textureDim>& coord,
            const isaac_vec_dim<T_n, T_CompType>& borderValue = isaac_vec_dim<T_n, T_CompType>(0)) const
        {
            isaac_vec_dim<T_n, isaac_float> result;
            if(T_filter == FilterType::LINEAR)
            {
                result = interpolate(texture, coord, borderValue);
            }
            else
            {
                result = isaac_vec_dim<T_n, isaac_float>(
                    safeMemoryAccess(texture, isaac_int_dim<T_textureDim>(coord), borderValue));
            }
            if(T_normalized)
                result /= isaac_float(std::numeric_limits<T_CompType>::max());
            return result;
        }


        template<typename T_Type, int T_textureDim, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE T_Type safeMemoryAccess(
            const Texture<T_Type, T_textureDim, T_indexType>& texture,
            const isaac_int_dim<T_textureDim>& coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = texture.getSizeWithGuard();
            const isaac_size_dim<T_textureDim> guardSize = texture.getGuardSize();

            isaac_int_dim<T_textureDim> offsetCoord;
            if(T_border == BorderType::REPEAT)
            {
                // Modulo modification to also account for negative values
                for(int i = 0; i < T_textureDim; ++i)
                {
                    offsetCoord[i] = (sizeWithGuard[i] + ((coord[i] + isaac_int(guardSize[i])) % sizeWithGuard[i]))
                        % sizeWithGuard[i];
                }
            }
            else if(T_border == BorderType::VALUE)
            {
                offsetCoord = coord + isaac_int_dim<T_textureDim>(guardSize);
                if(!isInLowerBounds(offsetCoord, isaac_int_dim<T_textureDim>(0))
                   || !isInUpperBounds(offsetCoord, isaac_int_dim<T_textureDim>(sizeWithGuard)))
                    return borderValue;
            }
            else
            {
                offsetCoord = glm::clamp(
                    coord + isaac_int_dim<T_textureDim>(guardSize),
                    isaac_int_dim<T_textureDim>(0),
                    isaac_int_dim<T_textureDim>(sizeWithGuard) - 1);
            }
            return texture[offsetCoord - isaac_int_dim<T_textureDim>(guardSize)];
        }

        template<typename T_Type, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_float interpolate(
            const Texture<T_Type, 2, T_indexType>& texture,
            isaac_float_dim<2> coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            coord -= isaac_float(0.5);
            isaac_float data4[2][2];
            for(int y = 0; y < 2; y++)
            {
                for(int x = 0; x < 2; x++)
                {
                    data4[x][y] = isaac_float(
                        safeMemoryAccess(texture, isaac_int2(glm::floor(coord)) + isaac_int2(x, y), borderValue));
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

        template<typename T_Type, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_float interpolate(
            const Texture<T_Type, 3, T_indexType>& texture,
            isaac_float_dim<3> coord,
            const T_Type& borderValue = T_Type(0)) const
        {
            coord -= isaac_float(0.5);
            isaac_float data8[2][2][2];
            for(int z = 0; z < 2; z++)
            {
                for(int y = 0; y < 2; y++)
                {
                    for(int x = 0; x < 2; x++)
                    {
                        data8[x][y][z] = isaac_float(safeMemoryAccess(
                            texture,
                            isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z),
                            borderValue));
                    }
                }
            }
            return trilinear(glm::fract(coord), data8);
        }

        template<int T_n, typename T_CompType, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_vec_dim<T_n, isaac_float> interpolate(
            const Texture<isaac_vec_dim<T_n, T_CompType>, 2, T_indexType>& texture,
            isaac_float_dim<2> coord,
            const isaac_vec_dim<T_n, T_CompType>& borderValue = isaac_vec_dim<T_n, T_CompType>(0)) const
        {
            coord -= isaac_float(0.5);
            isaac_vec_dim<T_n, isaac_float> data4[2][2];
            for(int y = 0; y < 2; y++)
            {
                for(int x = 0; x < 2; x++)
                {
                    data4[x][y] = isaac_vec_dim<T_n, isaac_float>(
                        safeMemoryAccess(texture, isaac_int2(glm::floor(coord)) + isaac_int2(x, y), borderValue));
                }
            }

            return bilinear(glm::fract(coord), data4);
        }

        template<int T_n, typename T_CompType, IndexType T_indexType>
        ISAAC_HOST_DEVICE_INLINE isaac_vec_dim<T_n, isaac_float> interpolate(
            const Texture<isaac_vec_dim<T_n, T_CompType>, 3, T_indexType>& texture,
            isaac_float_dim<3> coord,
            const isaac_vec_dim<T_n, T_CompType>& borderValue = isaac_vec_dim<T_n, T_CompType>(0)) const
        {
            coord -= isaac_float(0.5);
            isaac_vec_dim<T_n, isaac_float> data8[2][2][2];
            for(int z = 0; z < 2; z++)
            {
                for(int y = 0; y < 2; y++)
                {
                    for(int x = 0; x < 2; x++)
                    {
                        data8[x][y][z] = isaac_vec_dim<T_n, isaac_float>(safeMemoryAccess(
                            texture,
                            isaac_int3(glm::floor(coord)) + isaac_int3(x, y, z),
                            borderValue));
                    }
                }
            }


            return trilinear(glm::fract(coord), data8);
        }
    };


    /**
     * @brief Allocator class for textures
     *
     * @tparam T_DevAcc Alpaka device description for the buffer allocation
     * @tparam T_Type Type of the buffer values
     * @tparam T_textureDim Dimension of the Texture
     */
    template<typename T_DevAcc, typename T_Type, int T_textureDim, IndexType T_indexType = IndexType::SWEEP>
    class TextureAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        TextureAllocator(
            const T_DevAcc& devAcc,
            const isaac_size_dim<T_textureDim>& size,
            const isaac_size_dim<T_textureDim>& guardSize = isaac_size_dim<T_textureDim>(0))
            : bufferExtent(0)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            const isaac_size_dim<T_textureDim> sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;

            if(T_indexType == IndexType::MORTON)
            {
                ISAAC_IDX_TYPE maxDim = sizeWithGuard[0];
                std::cout << sizeWithGuard[0] << ", ";
                for(int i = 1; i < T_textureDim; ++i)
                {
                    std::cout << sizeWithGuard[i] << ", ";
                    maxDim = glm::max(maxDim, sizeWithGuard[i]);
                }
                bufferExtent = glm::pow(maxDim, ISAAC_IDX_TYPE(T_textureDim));
                std::cout << std::endl << bufferExtent << std::endl;
            }
            else
            {
                bufferExtent = sizeWithGuard[0];
                for(int i = 1; i < T_textureDim; ++i)
                {
                    bufferExtent *= (sizeWithGuard[i]);
                }
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);

            texture.init(alpaka::getPtrNative(buffer), size, guardSize);
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst) const
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        template<typename T_Queue, typename T_DstDev>
        void copyToTexture(T_Queue& queue, TextureAllocator<T_DstDev, T_Type, T_textureDim, T_indexType>& textureDst)
            const
        {
            assert(bufferExtent == textureDst.getBufferExtent());
            alpaka::memcpy(queue, textureDst.getTextureView(), buffer, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        Texture<T_Type, T_textureDim, T_indexType>& getTexture()
        {
            return texture;
        }

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>& getTextureView()
        {
            return buffer;
        }

        ISAAC_IDX_TYPE getBufferExtent()
        {
            return bufferExtent;
        }

    private:
        Texture<T_Type, T_textureDim, T_indexType> texture;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
    };


    // Texture class specialized for the communication of guard areas
    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    class SyncedTexture3DAllocator
    {
        using FraDim = alpaka::DimInt<1>;

    public:
        SyncedTexture3DAllocator(
            const T_DevAcc& devAcc,
            const isaac_size3& size,
            const isaac_size3& guardSize = isaac_size3(0))
            : bufferExtent(0)
            , buffer(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent))
        {
            const isaac_size3 sizeWithGuard = size + ISAAC_IDX_TYPE(2) * guardSize;

            ISAAC_IDX_TYPE totalAllocation = 0;
            if(T_indexType == IndexType::MORTON)
            {
                ISAAC_IDX_TYPE maxDim = sizeWithGuard[0];
                // std::cout << sizeWithGuard[0] << ", ";
                for(int i = 1; i < 3; ++i)
                {
                    std::cout << sizeWithGuard[i] << ", ";
                    maxDim = glm::max(maxDim, sizeWithGuard[i]);
                }
                bufferExtent = glm::pow(maxDim, ISAAC_IDX_TYPE(3));
                // std::cout << std::endl << bufferExtent << std::endl;
            }
            else
            {
                bufferExtent = sizeWithGuard[0];
                for(int i = 1; i < 3; ++i)
                {
                    bufferExtent *= (sizeWithGuard[i]);
                }
            }

            buffer = alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, bufferExtent);
            totalAllocation += bufferExtent;
            texture.init(alpaka::getPtrNative(buffer), size, guardSize);

            // std::cout << "Guard buffer allocations: " << std::endl;

            // start at index 1, because 0 is center
            for(ISAAC_IDX_TYPE i = 1; i < 27; ++i)
            {
                isaac_int3 direction = indexToDirection(i);
                isaac_size3 guardTexSize = isaac_size3(glm::abs(direction)) * guardSize;
                guardTexSize += (ISAAC_IDX_TYPE(1) - isaac_size3(glm::abs(direction))) * size;

                ISAAC_IDX_TYPE extent = guardTexSize.x * guardTexSize.y * guardTexSize.z;

                ownGuardBuffers.push_back(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, extent));
                totalAllocation += extent;
                ownGuardTextures.array[i].init(alpaka::getPtrNative(ownGuardBuffers.back()), guardTexSize);

                neighbourGuardBuffers.push_back(alpaka::allocBuf<T_Type, ISAAC_IDX_TYPE>(devAcc, extent));
                totalAllocation += extent;
                neighbourGuardTextures.array[i].init(alpaka::getPtrNative(neighbourGuardBuffers.back()), guardTexSize);

                /*
                std::cout << "side: (" << direction.x << ", " << direction.y << ", " << direction.z
                    << ")";
                std::cout << " size: (" << guardTexSize.x << ", " << guardTexSize.y << ", "
                    << guardTexSize.z << ")" << std::endl;
                */
            }
            totalAllocation *= sizeof(T_Type);
            // std::cout << "Total Allocation size: " << totalAllocation / float(1024 * 1024) << " MB" << std::endl;
        }

        template<typename T_Queue, typename T_ViewDst>
        void copyToBuffer(T_Queue& queue, T_ViewDst& viewDst) const
        {
            alpaka::memcpy(queue, viewDst, buffer, bufferExtent);
        }

        template<typename T_Queue, typename T_TextureAllocator>
        void copyToTexture(T_Queue& queue, T_TextureAllocator& textureDst) const
        {
            assert(bufferExtent == textureDst.getBufferExtent());
            alpaka::memcpy(queue, textureDst.getTextureView(), buffer, bufferExtent);
        }

        template<typename T_Queue>
        void clearColor(T_Queue& queue)
        {
            alpaka::memset(queue, buffer, 0, bufferExtent);
        }

        Texture<T_Type, 3, T_indexType>& getTexture()
        {
            return texture;
        }

        Texture<T_Type, 3>& getOwnGuardTexture(isaac_int3 direction)
        {
            return ownGuardTextures.get(direction);
        }

        Texture<T_Type, 3>& getOwnGuardTexture(isaac_uint index)
        {
            return ownGuardTextures.array[index];
        }

        Texture<T_Type, 3>& getNeighbourGuardTexture(isaac_int3 direction)
        {
            return neighbourGuardTextures.get(direction);
        }

        Texture<T_Type, 3>& getNeighbourGuardTexture(isaac_uint index)
        {
            return neighbourGuardTextures.array[index];
        }

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>& getTextureView()
        {
            return buffer;
        }

        ISAAC_IDX_TYPE getBufferExtent()
        {
            return bufferExtent;
        }

    private:
        Texture<T_Type, 3, T_indexType> texture;
        Neighbours<Texture<T_Type, 3>> ownGuardTextures;
        Neighbours<Texture<T_Type, 3>> neighbourGuardTextures;

        ISAAC_IDX_TYPE bufferExtent;

        alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE> buffer;
        std::vector<alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>> ownGuardBuffers;
        std::vector<alpaka::Buf<T_DevAcc, T_Type, FraDim, ISAAC_IDX_TYPE>> neighbourGuardBuffers;
    };


    template<typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Tex2D = Texture<T_Type, 2, T_indexType>;

    template<typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Tex3D = Texture<T_Type, 3, T_indexType>;

    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Tex2DAllocator = TextureAllocator<T_DevAcc, T_Type, 2, T_indexType>;

    template<typename T_DevAcc, typename T_Type, IndexType T_indexType = IndexType::SWEEP>
    using Tex3DAllocator = TextureAllocator<T_DevAcc, T_Type, 3, T_indexType>;


    template<int T_n>
    struct PersistentArrayStruct
    {
        Tex3D<isaac_float> textures[ZeroCheck<T_n>::value];
    };

    template<int T_n>
    struct AdvectionArrayStruct
    {
        Tex3D<isaac_byte> textures[ZeroCheck<T_n>::value];
    };

    struct GBuffer
    {
        isaac_size2 size;
        isaac_uint2 startOffset;
        Tex2D<isaac_byte4> color;
        Tex2D<isaac_float> depth;
        Tex2D<isaac_float3> normal;
        Tex2D<isaac_float> aoStrength;
    };

} // namespace isaac