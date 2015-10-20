/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ISAAC.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <stdlib.h>
#include <unistd.h>

/** This is a special list, where one (!) thread can push back while
 * one (!) can pop_front without any mutex needed.*/
template <typename T> class ThreadList
{
	public:
		typedef struct ThreadListContainer_struct
		{
			T t;
			bool deleted;
			struct ThreadListContainer_struct* next;
		} ThreadListContainer;
		typedef struct ThreadListContainer_struct *ThreadListContainer_ptr;
		ThreadList()
		{
			front = NULL;
			back = NULL;
			last_front_next = NULL;
		}
		void push_back(T t)
		{
			//Only this function / thread has the permission do add and remove elements
			ThreadListContainer_ptr ptr = (ThreadListContainer_ptr)malloc(sizeof(ThreadListContainer));
			ptr->t = t;
			ptr->next = NULL;
			ptr->deleted = false;
			if (back)
				back->next = ptr;
			else
				front = ptr;
			back = ptr;
			//Delete all as "deleted" marked elements from the front
			while (front && front->deleted)
			{
				ptr = front->next;
				free(front);
				front = ptr;
			}
			if (front == NULL)
				back = NULL;
		}
		T pop_front()
		{
			T t = NULL;
			ThreadListContainer_ptr new_front = last_front_next;
			if (new_front == NULL)
				new_front = front;
			while (new_front && new_front->deleted)
				new_front = new_front->next;
			if (new_front)
			{
				t = new_front->t;
				last_front_next = new_front->next;
				//After the next call push_back can remove "front"
				new_front->deleted = true;
			}
			return t;
		}
		void spin_over_delete()
		{
			//Busy waiting until all elements are deleted:
			while (front)
			{
				ThreadListContainer_ptr ptr;
				while (front && front->deleted)
				{
					ptr = front->next;
					free(front);
					front = ptr;
				}
				usleep(1);
			}
		}
		~ThreadList<T>()
		{
			spin_over_delete();
		}
		ThreadListContainer_ptr getFront()
		{
			return front;
		}
	private:
		ThreadListContainer_ptr front;
		ThreadListContainer_ptr last_front_next;
		ThreadListContainer_ptr back;
};
