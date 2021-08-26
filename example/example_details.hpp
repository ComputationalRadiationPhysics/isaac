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
 * You should have received a copy of the GNU General Lesser Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#include <fstream>
#include <isaac.hpp>
#include <sstream>
#include <string>

typedef float float3_t[3];


template<typename T_Stream, typename T_Host3, typename T_Dev3, typename T_Size>
void updateParticles(T_Stream stream, T_Host3 hostBuffer, T_Dev3 deviceBuffer, T_Size particle_count, float pos)
{
    for(ISAAC_IDX_TYPE i = 0; i < particle_count[0]; i++)
    {
        alpaka::getPtrNative(hostBuffer)[i].x = float((int(i * 29.6f)) % 64) / 64.0f;
        alpaka::getPtrNative(hostBuffer)[i].y = float((int(i * 23.1f)) % 64) / 64.0f;
        alpaka::getPtrNative(hostBuffer)[i].z
            = float((int((i * 7.9f + pos * (i % 20 + 1)) * 1000)) % 64000) / 1000.0f / 64.0f;
    }
    alpaka::memcpy(stream, deviceBuffer, hostBuffer, particle_count);
}

template<typename T_Stream, typename T_Host, typename T_Dev>
void updateVectorField(
    T_Stream stream,
    T_Host hostBuffer,
    T_Dev deviceBuffer,
    float pos,
    const isaac::isaac_size3& localSize,
    const isaac::isaac_size3& position,
    const isaac::isaac_size3& globalSize)
{
    using namespace isaac;
    float s = sin(pos * 10);
    for(ISAAC_IDX_TYPE x = 0; x < localSize.x; x++)
    {
        for(ISAAC_IDX_TYPE y = 0; y < localSize.y; y++)
        {
            for(ISAAC_IDX_TYPE z = 0; z < localSize.z; z++)
            {
                ISAAC_IDX_TYPE pos = x + y * localSize.x + z * localSize.x * localSize.y;
                isaac_size3 coord(x, y, z);
                isaac_float3 vector(
                    -(position.y + coord.y - globalSize.y * 0.5f) * s / isaac_float(globalSize.y),
                    (position.x + coord.x - globalSize.x * 0.5f) * s / isaac_float(globalSize.x),
                    0.3f);
                alpaka::getPtrNative(hostBuffer)[pos] = vector;
            }
        }
    }
    const alpaka::Vec<alpaka::DimInt<1>, ISAAC_IDX_TYPE> dataSize(localSize.x * localSize.y * localSize.z);
    alpaka::memcpy(stream, deviceBuffer, hostBuffer, dataSize);
}


template<typename T_Stream, typename T_Host1, typename T_Dev1, typename T_Host2, typename T_Dev2>
void updateData(
    T_Stream stream,
    T_Host1 hostBuffer1,
    T_Dev1 deviceBuffer1,
    T_Host2 hostBuffer2,
    T_Dev2 deviceBuffer2,
    size_t prod,
    float pos,
    const isaac::isaac_size3& localSize,
    const isaac::isaac_size3& position,
    const isaac::isaac_size3& globalSize)
{
    srand(0);
    float s = sin(pos);
    for(size_t x = 0; x < localSize[0]; x++)
    {
        for(size_t y = 0; y < localSize[1]; y++)
        {
            for(size_t z = 0; z < localSize[2]; z++)
            {
                float l_pos[3]
                    = {float(int(position[0]) + int(x) - int(globalSize[0]) / 2) / float(globalSize[0] / 2),
                       float(int(position[1]) + int(y) - int(globalSize[1]) / 2) / float(globalSize[1] / 2),
                       float(int(position[2]) + int(z) - int(globalSize[2]) / 2) / float(globalSize[2] / 2)};
                float l = sqrt(l_pos[0] * l_pos[0] + l_pos[1] * l_pos[1] + l_pos[2] * l_pos[2]);
                float intensity = 1.0f - l - float(rand() & ((2 << 16) - 1)) / float((2 << 17) - 1);
                intensity *= s + 1.5f;
                if(intensity < 0.0f)
                {
                    intensity = 0.0f;
                }
                if(intensity > 1.0f)
                {
                    intensity = 1.0f;
                }
                size_t pos = x + y * localSize[0] + z * localSize[0] * localSize[1];

                alpaka::getPtrNative(hostBuffer1)[pos].x = intensity;
                alpaka::getPtrNative(hostBuffer1)[pos].y = intensity;
                alpaka::getPtrNative(hostBuffer1)[pos].z = intensity;
                alpaka::getPtrNative(hostBuffer2)[pos] = (2.0f - l) * (2.0f - l) / 4.0f;
            }
        }
    }
    const alpaka::Vec<alpaka::DimInt<1>, ISAAC_IDX_TYPE> dataSize(
        ISAAC_IDX_TYPE(localSize[0]) * ISAAC_IDX_TYPE(localSize[1]) * ISAAC_IDX_TYPE(localSize[2]));
    alpaka::memcpy(stream, deviceBuffer1, hostBuffer1, dataSize);
    alpaka::memcpy(stream, deviceBuffer2, hostBuffer2, dataSize);
}


void mulToSmallestD(ISAAC_IDX_TYPE* d, int nr)
{
    if(d[0] < d[1]) // 0 < 1
    {
        if(d[2] < d[0])
        {
            d[2] *= nr; // 2 < 0 < 1
        }
        else
        {
            d[0] *= nr;
        } // 0 < 2 < 1 || 0 < 1 < 2
    }
    else // 1 < 0
    {
        if(d[2] < d[1])
        {
            d[2] *= nr; // 2 < 1 < 0
        }
        else
        {
            d[1] *= nr;
        } // 1 < 0 < 2 || 1 < 2 < 0
    }
}


void recursive_kgv(ISAAC_IDX_TYPE* d, int number, int test)
{
    if(number == 1)
    {
        return;
    }
    if(number == test)
    {
        mulToSmallestD(d, test);
        return;
    }
    if(number % test == 0)
    {
        number /= test;
        recursive_kgv(d, number, test);
        mulToSmallestD(d, test);
    }
    else
    {
        recursive_kgv(d, number, test + 1);
    }
}


template<
    typename T_Stream,
    typename T_Host1,
    typename T_Dev1,
    typename T_Host2,
    typename T_Dev2,
    typename T_Loc,
    typename T_Pos,
    typename T_Glo>
void read_vtk_to_memory(
    char* filename,
    T_Stream stream,
    T_Host1 hostBuffer1,
    T_Dev1 deviceBuffer1,
    T_Host2 hostBuffer2,
    T_Dev2 deviceBuffer2,
    size_t prod,
    float pos,
    T_Loc& localSize,
    T_Pos& position,
    T_Glo& globalSize,
    int& s_x,
    int& s_y,
    int& s_z)
{
    // Set first default values
    updateData(
        stream,
        hostBuffer1,
        deviceBuffer1,
        hostBuffer2,
        deviceBuffer2,
        prod,
        pos,
        localSize,
        position,
        globalSize);
    std::ifstream infile(filename);
    std::string line;
    // Format
    std::getline(infile, line);
    // Name
    std::getline(infile, line);
    printf("Reading data set %s\n", line.c_str());
    // Format
    std::getline(infile, line);
    if(line.compare(std::string("ASCII")) != 0)
    {
        printf("Only ASCII supported yet!\n");
        return;
    }
    // dataset
    std::getline(infile, line);
    if(line.compare(std::string("DATASET STRUCTURED_POINTS")) != 0)
    {
        printf("Only DATASET STRUCTURED_POINTS supported yet!\n");
        return;
    }
    // dimensions
    std::getline(infile, line);
    const char* buffer = line.c_str();
    int x, y, z;
    int i = strlen("DIMENSIONS ");
    x = atoi(&buffer[i]);
    while(buffer[i] && buffer[i] != ' ')
    {
        i++;
    }
    i++;
    y = atoi(&buffer[i]);
    while(buffer[i] && buffer[i] != ' ')
    {
        i++;
    }
    i++;
    z = atoi(&buffer[i]);
    printf("Dimensions: %i %i %i\n", x, y, z);
    // Spacing
    std::getline(infile, line);
    buffer = line.c_str();
    i = strlen("SPACING ");
    s_x = atoi(&buffer[i]);
    while(buffer[i] && buffer[i] != ' ')
    {
        i++;
    }
    i++;
    s_y = atoi(&buffer[i]);
    while(buffer[i] && buffer[i] != ' ')
    {
        i++;
    }
    i++;
    s_z = atoi(&buffer[i]);
    printf("Spacing: %i %i %i\n", s_x, s_y, s_z);
    if(size_t(x) != globalSize[0])
    {
        printf("Width needs to be %i instead of %i!\n", globalSize[0], x);
        return;
    }
    if(size_t(y) != globalSize[1])
    {
        printf("Width needs to be %i instead of %i!\n", globalSize[1], y);
        return;
    }
    if(size_t(z) != globalSize[2])
    {
        printf("Width needs to be %i instead of %i!\n", globalSize[2], z);
        return;
    }
    // ORIGIN, POINT_DATA, SCALARS, LOOKUP_TABLE
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    x = 0;
    y = 0;
    z = 0;
    while(std::getline(infile, line))
    {
        char* buffer = const_cast<char*>(line.c_str());
        while(buffer[0] && buffer[0] != '\n')
        {
            int value = strtol(buffer, &buffer, 0);
            ;
            int t_x = x - position[0];
            int t_y = y - position[1];
            int t_z = z - position[2];
            if(t_x >= 0 && size_t(t_x) < localSize[0] && t_y >= 0 && size_t(t_y) < localSize[1] && t_z >= 0
               && size_t(t_z) < localSize[2])
            {
                size_t pos = t_x + t_y * localSize[0] + t_z * localSize[0] * localSize[1];

                alpaka::getPtrNative(hostBuffer2)[pos] = (float) value;
            }
            x++;
            if(size_t(x) >= globalSize[0])
            {
                x = 0;
                y++;
                if(size_t(y) >= globalSize[1])
                {
                    y = 0;
                    z++;
                }
            }
        }
    }

    const alpaka::Vec<alpaka::DimInt<1>, ISAAC_IDX_TYPE> dataSize(
        ISAAC_IDX_TYPE(localSize[0]) * ISAAC_IDX_TYPE(localSize[1]) * ISAAC_IDX_TYPE(localSize[2]));
    alpaka::memcpy(stream, deviceBuffer2, hostBuffer2, dataSize);
}
