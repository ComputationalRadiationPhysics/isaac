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

//#define SEND_PARTICLES

#include "isaac.hpp"
#include <IceT.h>
//Against annoying C++11 warning in mpi.h
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wall"
#include <IceTMPI.h>
#pragma GCC diagnostic pop

#define MASTER_RANK 0

//#define SEND_PARTICLES

#ifdef SEND_PARTICLES
	#define PARTICLES_PER_NODE 8
#endif

void recursive_kgv(size_t* d,int number,int test);

int main(int argc, char **argv)
{
	char __server[] = "localhost";
	char* server = __server;
	if (argc > 1)
		server = argv[1];
	int port = 2460;
	if (argc > 2)
		port = atoi(argv[2]);

	//MPI Init
	int rank,numProc,provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);

	//Let's calculate the best spatial distribution of the dimensions so that d[0]*d[1]*d[2] = numProc
	size_t d[3] = {1,1,1};
	recursive_kgv(d,numProc,2);

	size_t p[3] = { rank % d[0], (rank / d[0]) % d[1],  (rank / d[0] / d[1]) % d[2] };
	
	//Let's use this to create some random particles inside my box
	#ifdef SEND_PARTICLES

		//With this I can calculate my box position and size
		float box_size[3];
		for (int i = 0; i < 3; i++)
			box_size[i] = 1.0f/float(d[i])*2.0f;
		float box_position[3];
		for (int i = 0; i < 3; i++)
			box_position[i] = (float)p[i]*box_size[i]-1.0f;

		srand(rank*time(NULL));
		float particles[PARTICLES_PER_NODE][3];
		float forces[PARTICLES_PER_NODE][3];
		for (int i = 0; i < PARTICLES_PER_NODE; i++)
			for (int j = 0; j < 3; j++)
			{
				particles[i][j] = box_position[j] + (float)rand() / (float)RAND_MAX * box_size[j];
				forces[i][j] = 0.0f;
			}
	#endif
	//Let's generate some unique name for the simulation and broadcast it
	int id;
	if (rank == MASTER_RANK)
	{
		srand(time(NULL));
		id = rand() % 1000000;
	}
	MPI_Bcast(&id,sizeof(id), MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
	char name[32];
	sprintf(name,"Example_%i",id);
	printf("Using name %s\n",name);
	
	typedef float float3_t[3];
	//int width = 1920;
	//int height = 1080;
	int width = 1024;
	int height = 768;
	//int width = 800;
	//int height = 600;
	
	#ifdef ISAAC_ALPAKA
		//Now we initialize the Isaac Insitu Plugin with the name, the number of the master, the server, it's IP, the count of framebuffer to be created and the size per framebuffer
		using AccDim = alpaka::dim::DimInt<3>;
		using SimDim = alpaka::dim::DimInt<3>;
		//using Acc = alpaka::acc::AccGpuCudaRt<AccDim, size_t>;
		//using Stream  = alpaka::stream::StreamCudaRtSync;
		using Acc = alpaka::acc::AccCpuOmp2Blocks<AccDim, size_t>;
		using Stream  = alpaka::stream::StreamCpuSync;
		//using Acc = alpaka::acc::AccCpuOmp2Threads<AccDim, size_t>;
		//using Stream  = alpaka::stream::StreamCpuSync;
		
		/*if ( boost::mpl::not_<boost::is_same<Acc, alpaka::acc::AccGpuCudaRt<AccDim, size_t> > >::value )
		{
			width /= 2;
			height /= 2;
		}*/
		
		using DevAcc = alpaka::dev::Dev<Acc>;
		using DevHost = alpaka::dev::DevCpu;
		
		DevAcc  devAcc  (alpaka::dev::DevMan<Acc>::getDevByIdx(0));
		DevHost devHost (alpaka::dev::cpu::getDev());
		Stream  stream  (devAcc);
		
		const alpaka::Vec<SimDim, size_t> global_size(d[0]*64,d[1]*64,d[2]*64);
		const alpaka::Vec<SimDim, size_t> local_size(size_t(64),size_t(64),size_t(64));
		const alpaka::Vec<SimDim, size_t> position(p[0]*64,p[1]*64,p[2]*64);
		IsaacVisualization<DevHost,Acc,Stream,AccDim,SimDim> visualization(devHost,devAcc,stream,name,MASTER_RANK,server,port,width,height,global_size,local_size,position);
	#else
		typedef boost::mpl::int_<3> SimDim;
		std::vector<size_t> global_size;
			global_size.push_back(d[0]*64);
			global_size.push_back(d[1]*64);
			global_size.push_back(d[2]*64);
		std::vector<size_t> local_size;
			local_size.push_back(64);
			local_size.push_back(64);
			local_size.push_back(64);
		std::vector<size_t> position;
			position.push_back(p[0]*64);
			position.push_back(p[1]*64);
			position.push_back(p[2]*64);
		IsaacVisualization<SimDim> visualization(name,MASTER_RANK,server,port,width,height,global_size,local_size,position);
	#endif
	
	//Init Device memory and adding source:
	#ifdef ISAAC_ALPAKA
		alpaka::mem::buf::Buf<DevHost, float3_t, SimDim, size_t> hostBuffer1   ( alpaka::mem::buf::alloc<float3_t, size_t>(devHost, local_size));
		alpaka::mem::buf::Buf<DevAcc, float3_t, SimDim, size_t>  deviceBuffer1 ( alpaka::mem::buf::alloc<float3_t, size_t>(devAcc,  local_size));
		alpaka::mem::buf::Buf<DevHost, float, SimDim, size_t> hostBuffer2   ( alpaka::mem::buf::alloc<float, size_t>(devHost, local_size));
		alpaka::mem::buf::Buf<DevAcc, float, SimDim, size_t>  deviceBuffer2 ( alpaka::mem::buf::alloc<float, size_t>(devAcc,  local_size));
		for (size_t x = 0; x < local_size[0]; x++)
			for (size_t y = 0; y < local_size[1]; y++)
				for (size_t z = 0; z < local_size[2]; z++)
				{
					size_t pos = x + y * local_size[0] + z * local_size[0] * local_size[1];
					alpaka::mem::view::getPtrNative(hostBuffer1)[pos][0] = 1.0f;
					alpaka::mem::view::getPtrNative(hostBuffer1)[pos][1] = (float)(rank+1)/(float)numProc;
					alpaka::mem::view::getPtrNative(hostBuffer1)[pos][2] = 1.0f-(float)(rank+1)/(float)numProc;
					alpaka::mem::view::getPtrNative(hostBuffer2)[pos] = (float)(rank+1)/(float)numProc;
				}
		alpaka::mem::view::copy(stream, deviceBuffer1, hostBuffer1, local_size);
		alpaka::mem::view::copy(stream, deviceBuffer2, hostBuffer2, local_size);
		
		visualization.registerSource("source1",reinterpret_cast<float*>(alpaka::mem::view::getPtrNative(deviceBuffer1)),3);
		visualization.registerSource("source2",reinterpret_cast<float*>(alpaka::mem::view::getPtrNative(deviceBuffer2)),1);
	#else
		size_t prod = local_size[0]*local_size[1]*local_size[2];
		float3_t* hostBuffer1 = (float3_t*)malloc(sizeof(float3_t)*prod);
		float3_t* deviceBuffer1; cudaMalloc((float3_t**)&deviceBuffer1, sizeof(float3_t)*prod);
		float* hostBuffer2 = (float*)malloc(sizeof(float)*prod);
		float* deviceBuffer2; cudaMalloc((float**)&deviceBuffer2, sizeof(float)*prod);
		for (size_t x = 0; x < local_size[0]; x++)
			for (size_t y = 0; y < local_size[1]; y++)
				for (size_t z = 0; z < local_size[2]; z++)
				{
					size_t pos = x + y * local_size[0] + z * local_size[0] * local_size[1];
					hostBuffer1[pos][0] = 1.0f;
					hostBuffer1[pos][1] = (float)(rank+1)/(float)numProc;
					hostBuffer1[pos][2] = 1.0f-(float)(rank+1)/(float)numProc;
					hostBuffer2[pos] = (float)(rank+1)/(float)numProc;
				}
		cudaMemcpy(deviceBuffer1, hostBuffer1, sizeof(float3_t)*prod, cudaMemcpyHostToDevice);
		cudaMemcpy(deviceBuffer2, hostBuffer2, sizeof(float)*prod, cudaMemcpyHostToDevice);
		
		visualization.registerSource("source1",(float*)deviceBuffer1,3);
		visualization.registerSource("source2",deviceBuffer2,1);
	#endif
	//Setting up the metadata description (only master, but however slaves could then metadata metadata, too, it would be merged)
	if (rank == MASTER_RANK)
	{
		json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_string( "Engery in kJ" ) );
		json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_string( "Speed in multiplies of the speed of a hare" ) );
		#ifdef SEND_PARTICLES
			json_t *particle_array = json_array();
			json_object_set_new( visualization.getJsonMetaRoot(), "reference particles", particle_array );
			json_array_append_new( particle_array, json_string( "X" ) );
			json_array_append_new( particle_array, json_string( "Y" ) );
			json_array_append_new( particle_array, json_string( "Z" ) );
		#endif
	}

	//finish init and sending the meta data scription to the isaac server
	if (visualization.init())
	{
		fprintf(stderr,"Isaac init failed.\n");
		return -1;
	}
	
	float a = 0.0f;
	volatile int force_exit = 0;
	int start = visualization.getTicksUs();
	int count = 0;
	int drawing_time = 0;
	while (!force_exit)
	{
		a += 0.01f;
		//Every frame we fill the metadata with data instead of descriptions
		if (rank == MASTER_RANK)
		{
			json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_real( a ) );
			json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_real( a*a ) );
		}
		#ifdef SEND_PARTICLES
			//every thread fills "his" particles
			json_t *particle_array = json_array();
			for (int i = 0; i < PARTICLES_PER_NODE; i++)
			{
				json_t *position = json_array();
				json_array_append_new( particle_array, position );
				//Recalculate force based on distance to box center and add it
				for (int j = 0; j < 3; j++)
				{
					float distance = (box_position[j] + box_size[j] / 2.0f) - particles[i][j];
					forces[i][j] += distance / 10000.0f;
					particles[i][j] += forces[i][j];
					json_array_append_new( position, json_real( particles[i][j] ) );
				}
			}
			json_object_set_new( visualization.getJsonMetaRoot(), "reference particles", particle_array );
		#endif
		//Visualize and send data to the server
		int start_drawing = visualization.getTicksUs();
		#ifdef SEND_PARTICLES
			json_t* meta = visualization.doVisualization(META_MERGE);
		#else
			json_t* meta = visualization.doVisualization(META_MASTER);
		#endif
		drawing_time +=visualization.getTicksUs() - start_drawing;
		//New metadata from the server?
		if (meta)
		{
			//Let's print it to stdout
			char* buffer = json_dumps( meta, 0 );
			printf("META (%i): %s\n",rank,buffer);
			free(buffer);
			//And let's also check for an exit message
			if ( rank == MASTER_RANK && json_integer_value( json_object_get(meta, "exit") ) )
				force_exit = 1;
			//Deref the jansson json root! Otherwise we would get a memory leak
			json_decref( meta );
		}
		//printf("%i: Sent dummy meta data\n",rank);
		//sync
		MPI_Bcast((void*)&force_exit,sizeof(force_exit), MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
		usleep(100);
		count++;
		if (rank == MASTER_RANK)
		{
			int end = visualization.getTicksUs();
			int diff = end-start;
			if (diff >= 1000000)
			{
				visualization.merge_time -= visualization.kernel_time + visualization.copy_time;
				printf("FPS: %.1f \n\tDrawing: %.1f ms\n\t\tSorting: %.1f ms\n\t\tMerge: %.1f ms\n\t\tKernel: %.1f ms\n\t\tCopy: %.1f ms\n\t\tVideo: %.1f ms\n",
					(float)count*1000000.0f/(float)diff,
					(float)drawing_time/1000.0f/(float)count,
					(float)visualization.sorting_time/1000.0f/(float)count,
					(float)visualization.merge_time/1000.0f/(float)count,
					(float)visualization.kernel_time/1000.0f/(float)count,
					(float)visualization.copy_time/1000.0f/(float)count,
					(float)visualization.video_send_time/1000.0f/(float)count);
				visualization.sorting_time = 0;
				visualization.merge_time = 0;
				visualization.kernel_time = 0;
				visualization.copy_time = 0;
				visualization.video_send_time = 0;
				start = end;
				count = 0;
				drawing_time = 0;
			}
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	printf("%i finished\n",rank);
	
	#ifndef ISAAC_ALPAKA
		free(hostBuffer1);
		free(hostBuffer2);
		cudaFree(deviceBuffer1);
		cudaFree(deviceBuffer2);
	#endif
	
	MPI_Finalize();
	return 0;
}

// Not necessary, just for the example

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
