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
 
#include "isaac.hpp"
#include <IceT.h>
//Against annoying C++11 warning in mpi.h
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wall"
#include <IceTMPI.h>
#pragma GCC diagnostic pop

#define MASTER_RANK 0
#define PARTICLES_PER_NODE 8

void recursive_kgv(int* d,int number,int test);

int main(int argc, char **argv)
{
	//MPI Init
	int rank,numProc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);
	
	//Let's calculate the best spatial distribution of the dimensions so that d[0]*d[1]*d[2] = numProc
	int d[3] = {1,1,1};
	recursive_kgv(d,numProc,2);
	
	//With this I can calculate my box position and size
	float box_size[3];
	for (int i = 0; i < 3; i++)
		box_size[i] = 1.0f/float(d[i])*2.0f;
	float box_position[3];
	int p[3] = { rank % d[0], (rank / d[0]) % d[1],  (rank / d[0] / d[1]) % d[2] };
	for (int i = 0; i < 3; i++)
		box_position[i] = (float)p[i]*box_size[i]-1.0f;

	//Let's use this to create some random particles inside my box
	srand(rank*time(NULL));	
	float particles[PARTICLES_PER_NODE][3];
	float forces[PARTICLES_PER_NODE][3];
	for (int i = 0; i < PARTICLES_PER_NODE; i++)
		for (int j = 0; j < 3; j++)
		{
			particles[i][j] = box_position[j] + (float)rand() / (float)RAND_MAX * box_size[j];
			forces[i][j] = 0.0f;
		}

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
	
	//Now we initialize the Isaac Insitu Plugin with the name, the number of the master, the server, it's IP, the count of framebuffer to be created and the size per framebuffer
	IsaacVisualization visualization = IsaacVisualization(name,MASTER_RANK,"localhost",2460,3,512);
	
	//Setting up the metadata description (only master, but however slaves could then metadata metadata, too, it would be merged)
	if (rank == MASTER_RANK)
	{
		json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_string( "Engery in kJ" ) );
		json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_string( "Speed in multiplies of the speed of a hare" ) );
		json_t *particle_array = json_array();
		json_object_set_new( visualization.getJsonMetaRoot(), "reference particles", particle_array );
		json_array_append_new( particle_array, json_string( "X" ) );
		json_array_append_new( particle_array, json_string( "Y" ) );
		json_array_append_new( particle_array, json_string( "Z" ) );
	}

	//finish init and sending the meta data scription to the isaac server
	if (visualization.init())
	{
		fprintf(stderr,"Isaac init failed.\n");
		return -1;
	}
	
	float a = 0.0f;
	volatile int force_exit = 0;
	while (!force_exit)
	{
		a += 0.01f;
		//New metadata from the server?
		while (json_t* meta = visualization.getMeta())
		{
			char* buffer = json_dumps( meta, 0 );
			//Let's print it to stdout
			printf("META (%i): %s\n",rank,buffer);
			//And let's also check for an exit message
			if (rank == MASTER_RANK && json_integer_value( json_object_get(meta, "exit") ))
				force_exit = 1;
			//Free the buffer from jansson
			free(buffer);
			//Deref the jansson json root! Otherwise we would get a memory leak
			json_decref( meta );
		}
		//Every frame we fill the metadata with data instead of descriptions
		if (rank == MASTER_RANK)
		{
			json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_real( a ) );
			json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_real( a*a ) );
		}
		//every thread fills "his" particles
		if (rank % 2 == 0)
		{
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
		}
		//Visualize and send data to the server
		visualization.doVisualization((rank % 2 == 0)?META_MERGE:META_NONE,(numProc+1)/2);
		//printf("%i: Sent dummy meta data\n",rank);
		//sync
		MPI_Bcast((void*)&force_exit,sizeof(force_exit), MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
		usleep(50000);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	printf("%i finished\n",rank);
	MPI_Finalize();
	return 0;
}

// Not necessary, just for the example

void mul_to_smallest_d(int *d,int nr)
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

void recursive_kgv(int* d,int number,int test)
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

