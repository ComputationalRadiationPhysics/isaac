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
#include <signal.h>
#include <IceT.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wall"
#include <IceTMPI.h>
#pragma GCC diagnostic pop

volatile sig_atomic_t force_exit = 0;

void sighandler(int sig)
{
	printf("\n");
	force_exit = 1;
}

int main(int argc, char **argv)
{
	IsaacVisualization visualization = IsaacVisualization("Example","localhost",2460,3,512);
	//Setting up the metadata description
	json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_string( "Engery in kJ" ) );
	json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_string( "Speed in multiplies of the speed of a hare" ) );
	json_t *particle_array = json_array();
	for (int i = 0; i < 5; i++)
	{
		char buffer[256];
		sprintf(buffer,"Reference Particle %i",i);
		json_array_append( particle_array, json_string( buffer ) );
	}
	json_object_set_new( visualization.getJsonMetaRoot(), "reference particles", particle_array );
	//finish init and sending the meta data scription
	if (visualization.init())
	{
		fprintf(stderr,"Isaac init failed.\n");
		return -1;
	}
	signal(SIGINT, sighandler);
	float a = 0.0f;
	int c = 0;
	while (!force_exit)
	{
		c++;
		a += 0.01f;
		//New metadata for me?
		while (json_t* meta = visualization.getMeta())
		{
			char* buffer = json_dumps( meta, 0 );
			printf("META: %s\n",buffer);
			free(buffer);
			json_decref( meta );
		}
		//Setting the metadata values described earlier
		json_object_set_new( visualization.getJsonMetaRoot(), "energy", json_real( a ) );
		json_object_set_new( visualization.getJsonMetaRoot(), "speed", json_real( c ) );
		json_t *particle_array = json_array();
		for (int i = 0; i < 5; i++)
			json_array_append( particle_array, json_integer(rand()%100) );
		json_object_set_new( visualization.getJsonMetaRoot(), "reference particles", particle_array );
		//Visualize
		visualization.doVisualization();
		printf("Sent dummy meta data\n");
		usleep(500000);
	}
	printf("%i\n",c);
	signal(SIGINT, SIG_DFL);
	return 0;
}
