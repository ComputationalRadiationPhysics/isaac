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
#include <sstream>
#include <string>

typedef float float3_t[3];

template <
	typename TStream,
	typename THost1,
	typename TDev1,
	typename THost2,
	typename TDev2,
	typename TLoc,
	typename TPos,
	typename TGlo
>
void update_data(
	TStream stream,
	THost1 hostBuffer1,
	TDev1 deviceBuffer1,
	THost2 hostBuffer2,
	TDev2 deviceBuffer2,
	size_t prod,
	float pos,
	TLoc& local_size,
	TPos& position,
	TGlo& global_size)
{
	srand(0);
	float s = sin(pos);
	for (size_t x = 0; x < local_size[0]; x++)
		for (size_t y = 0; y < local_size[1]; y++)
			for (size_t z = 0; z < local_size[2]; z++)
			{
				float l_pos[3] =
				{
					float(int(position[0]) + int(x) - int(global_size[0]) / 2) / float( global_size[0] / 2),
					float(int(position[1]) + int(y) - int(global_size[1]) / 2) / float( global_size[1] / 2),
					float(int(position[2]) + int(z) - int(global_size[2]) / 2) / float( global_size[2] / 2)
				};
				float l = sqrt( l_pos[0] * l_pos[0] + l_pos[1] * l_pos[1] + l_pos[2] * l_pos[2] );
				float intensity = 1.0f - l - float(rand() & ((2 << 16) - 1))/float( (2 << 17) - 1 );
				intensity *= s+1.5f;
				if (intensity < 0.0f)
					intensity = 0.0f;
				if (intensity > 1.0f)
					intensity = 1.0f;
				size_t pos = x + y * local_size[0] + z * local_size[0] * local_size[1];
#if ISAAC_ALPAKA == 1
				alpaka::mem::view::getPtrNative(hostBuffer1)[pos][0] = intensity;
				alpaka::mem::view::getPtrNative(hostBuffer1)[pos][1] = intensity;
				alpaka::mem::view::getPtrNative(hostBuffer1)[pos][2] = intensity;
				alpaka::mem::view::getPtrNative(hostBuffer2)[pos] =  (2.0f - l)*(2.0f - l) / 4.0f;
			}
	const alpaka::vec::Vec<alpaka::dim::DimInt<1>, size_t> data_size(size_t(local_size[0]) * size_t(local_size[1]) * size_t(local_size[2]));
	alpaka::mem::view::copy(stream, deviceBuffer1, hostBuffer1, data_size);
	alpaka::mem::view::copy(stream, deviceBuffer2, hostBuffer2, data_size);
#else
				hostBuffer1[pos][0] = intensity;
				hostBuffer1[pos][1] = intensity;
				hostBuffer1[pos][2] = intensity;
				hostBuffer2[pos] = (2.0f - l)*(2.0f - l) / 4.0f;
			}
	cudaMemcpy(deviceBuffer1, hostBuffer1, sizeof(float3_t)*prod, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBuffer2, hostBuffer2, sizeof(float)*prod, cudaMemcpyHostToDevice);
#endif
}

void mul_to_smallest_d(size_t *d,int nr)
{
	if (d[0] < d[1]) // 0 < 1
	{
		if (d[2] < d[0])
			d[2] *= nr; //2 < 0 < 1
		else
			d[0] *= nr; //0 < 2 < 1 || 0 < 1 < 2
	}
	else // 1 < 0
	{
		if (d[2] < d[1])
			d[2] *= nr; // 2 < 1 < 0
		else
			d[1] *= nr; // 1 < 0 < 2 || 1 < 2 < 0
	}
}

void recursive_kgv(size_t* d,int number,int test)
{
	if (number == 1)
		return;
	if (number == test)
	{
		mul_to_smallest_d(d,test);
		return;
	}
	if (number % test == 0)
	{
		number /= test;
		recursive_kgv(d,number,test);
		mul_to_smallest_d(d,test);
	}
	else
		recursive_kgv(d,number,test+1);
}

template <
	typename TStream,
	typename THost1,
	typename TDev1,
	typename THost2,
	typename TDev2,
	typename TLoc,
	typename TPos,
	typename TGlo
>
void read_vtk_to_memory(
	char* filename,
	TStream stream,
	THost1 hostBuffer1,
	TDev1 deviceBuffer1,
	THost2 hostBuffer2,
	TDev2 deviceBuffer2,
	size_t prod,
	float pos,
	TLoc& local_size,
	TPos& position,
	TGlo& global_size,
	int& s_x,
	int& s_y,
	int& s_z)
{
	//Set first default values
	update_data(stream,hostBuffer1,deviceBuffer1,hostBuffer2,deviceBuffer2,prod,pos,local_size,position,global_size);
	std::ifstream infile( filename );
	std::string line;
	//Format
	std::getline(infile, line);
	//Name
	std::getline(infile, line);
	printf("Reading data set %s\n",line.c_str());
	//Format
	std::getline(infile, line);
	if ( line.compare(std::string("ASCII")) != 0)
	{
		printf("Only ASCII supported yet!\n");
		return;
	}
	//dataset
	std::getline(infile, line);
	if ( line.compare(std::string("DATASET STRUCTURED_POINTS")) != 0)
	{
		printf("Only DATASET STRUCTURED_POINTS supported yet!\n");
		return;
	}
	//dimensions
	std::getline(infile, line);
	const char* buffer = line.c_str();
	int x,y,z;
	int i = strlen("DIMENSIONS ");
	x = atoi(&buffer[i]);
	while (buffer[i] && buffer[i] != ' ' )
		i++;
	i++;
	y = atoi(&buffer[i]);
	while (buffer[i] && buffer[i] != ' ' )
		i++;
	i++;
	z = atoi(&buffer[i]);
	printf("Dimensions: %i %i %i\n",x,y,z);
	//Spacing
	std::getline(infile, line);
	buffer = line.c_str();
	i = strlen("SPACING ");
	s_x = atoi(&buffer[i]);
	while (buffer[i] && buffer[i] != ' ' )
		i++;
	i++;
	s_y = atoi(&buffer[i]);
	while (buffer[i] && buffer[i] != ' ' )
		i++;
	i++;
	s_z = atoi(&buffer[i]);
	printf("Spacing: %i %i %i\n",s_x,s_y,s_z);
	if (size_t(x) != global_size[0])
	{
		printf("Width needs to be %lu instead of %i!\n",global_size[0],x);
		return;
	}
	if (size_t(y) != global_size[1])
	{
		printf("Width needs to be %lu instead of %i!\n",global_size[1],y);
		return;
	}
	if (size_t(z) != global_size[2])
	{
		printf("Width needs to be %lu instead of %i!\n",global_size[2],z);
		return;
	}
	//ORIGIN, POINT_DATA, SCALARS, LOOKUP_TABLE
	std::getline(infile, line);
	std::getline(infile, line);
	std::getline(infile, line);
	std::getline(infile, line);
	x = 0;
	y = 0;
	z = 0;
	while (std::getline(infile, line))
	{
		char* buffer = const_cast<char*>(line.c_str());
		while (buffer[0] && buffer[0] != '\n')
		{
			int value = strtol (buffer,&buffer,0);;
			int t_x = x - position[0];
			int t_y = y - position[1];
			int t_z = z - position[2];
			if (t_x >= 0 && size_t(t_x) < local_size[0] &&
				t_y >= 0 && size_t(t_y) < local_size[1] &&
				t_z >= 0 && size_t(t_z) < local_size[2])
			{
				size_t pos = t_x + t_y * local_size[0] + t_z * local_size[0] * local_size[1];
				#if ISAAC_ALPAKA == 1
					alpaka::mem::view::getPtrNative(hostBuffer2)[pos] = (float)value;
				#else
					hostBuffer2[pos] = (float)value;
				#endif
			}
			x++;
			if (size_t(x) >= global_size[0])
			{
				x = 0;
				y++;
				if (size_t(y) >= global_size[1])
				{
					y = 0;
					z++;
				}
			}
		}
	}

#if ISAAC_ALPAKA == 1
	const alpaka::vec::Vec<alpaka::dim::DimInt<1>, size_t> data_size(size_t(local_size[0]) * size_t(local_size[1]) * size_t(local_size[2]));
	alpaka::mem::view::copy(stream, deviceBuffer2, hostBuffer2, data_size);
#else
	cudaMemcpy(deviceBuffer2, hostBuffer2, sizeof(float)*prod, cudaMemcpyHostToDevice);
#endif
}
