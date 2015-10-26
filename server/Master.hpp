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

#ifndef __MASTER
#define __MASTER

#include <string>
#include <list>
#include "Common.hpp"
#include "MetaDataConnector.hpp"
class MetaDataConnector;

#include <signal.h>
#include "MetaDataClient.hpp"
#include "InsituConnectorMaster.hpp"
#include <memory>

typedef struct MetaDataConnectorContainer_struct
{
	MetaDataConnector* connector;
	pthread_t thread;
} MetaDataConnectorContainer;

#define MAX_NODES 999999

class InsituConnectorGroup
{
	public:
		InsituConnectorGroup(std::string name)
		{
			this->name = name;
			this->nodes = MAX_NODES;
			this->id = 0;
			this->merge_count = 0;
			this->meta_merge_count = 0;
			this->initData = json_object();
			json_object_set_new( initData, "type", json_string( "tell plugin" ) );
			//name will be merged from mergeJSON(json_t* candidate)
			//json_object_set_new( json_root, "name", json_string( name.c_str() ) );
		}
		int getID()
		{
			return id;
		}
		void mergeJSON(json_t* result,json_t* candidate,InsituConnectorContainer* myself)
		{
			const char *c_key;
			const char *r_key;
			json_t *c_value;
			json_t *r_value;
			//metadata merge, old values stay, arrays are merged
			json_t* m_candidate = json_object_get(candidate, "metadata");
			json_t* m_result = json_object_get(result, "metadata");
			void *temp,*temp2;
			if (m_candidate && m_result)
			{
				json_object_foreach_safe( m_candidate, temp, c_key, c_value )
				{
					bool found = false;
					json_object_foreach_safe( m_result, temp2, r_key, r_value )
					{
						if (strcmp(r_key,c_key) == 0)
						{
							if (json_is_array(r_value) && json_is_array(c_value))
								json_array_extend(r_value,c_value);
							found = true;
							break;
						}
					}
					if (!found)
						json_object_set( m_result, c_key, c_value );
				}
			}			
			//general merge, old values stay
			json_object_foreach_safe( candidate, temp, c_key, c_value )
			{
				bool found = false;
				json_object_foreach_safe( result, temp2, r_key, r_value )
				{
					if (strcmp(r_key,c_key) == 0)
					{
						found = true;
						break;
					}
				}
				if (!found)
				{
					json_object_set( result, c_key, c_value );
					if (myself && nodes == MAX_NODES && strcmp(c_key, "nodes" ) == 0)
					{
						nodes = json_integer_value ( c_value );
						//This must be the result, so set it:
						master = myself;
						id = master->connector->getID();
					}
				}
			}
		}
		ThreadList< InsituConnectorContainer* > elements;
		int nodes;
		std::string name;
		json_t* initData;
		InsituConnectorContainer* master;
		int id;
		int merge_count;
		int meta_merge_count;
		json_t* mergeData;
		int merge_count_max;
};

class Master
{
	public:
		Master(std::string name,int inner_port);
		~Master();
		errorCode addDataConnector(MetaDataConnector *dataConnector);
		MetaDataClient* addDataClient();
		errorCode run();
		static volatile sig_atomic_t force_exit;
	private:
		InsituConnectorMaster insituMaster;
		json_t* masterHello;
		std::string name;
		std::list< MetaDataConnectorContainer > dataConnectorList;
		ThreadList< InsituConnectorGroup* > insituConnectorGroupList;
		ThreadList< MetaDataClient* > dataClientList;
		int inner_port;
		pthread_t insituThread;
};

#endif
