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

#pragma once

#include <stdlib.h>
#include <unistd.h>

/** This is a special list, where one (!) thread can push back while
 * one (!) can pop_front with only one mutex needed and (hopefully)
 * not too much busy waiting
 * However there are some rules:
 * - As said: Only one thread can only pop_front and remove
 * - As said: Only one thread can use push_back
 * - Only the thread, which uses pop_front, may also iterate over the list
 * - Iterating should only use getFront and ->next */
template <typename T> class ThreadList
{
	public:
		typedef struct ThreadListContainer_struct
		{
			T t;
			struct ThreadListContainer_struct* next;
		} ThreadListContainer;
		typedef struct ThreadListContainer_struct *ThreadListContainer_ptr;
		ThreadList()
		{
			pthread_mutex_init (&remove_mutex, NULL);
			front = NULL;
			back = NULL;
			l = 0;
		}
		void push_back(T t)
		{
			//Only this function / thread has the permission do add and remove elements
			ThreadListContainer_ptr ptr = (ThreadListContainer_ptr)malloc(sizeof(ThreadListContainer));
			ptr->t = t;
			ptr->next = NULL;
			pthread_mutex_lock (&remove_mutex);
			if (back)
				back->next = ptr;
			else
				front = ptr;
			back = ptr;
			l++;
			pthread_mutex_unlock (&remove_mutex);
		}
		T pop_front()
		{
			T t = NULL;
			if (front)
			{
				t = front->t;
				pthread_mutex_lock (&remove_mutex);
				//delete front
				ThreadListContainer_ptr	next = front->next;
				free(front);
				front = next;
				if (front == NULL)
					back = NULL;
				l--;
				pthread_mutex_unlock (&remove_mutex);
			}
			return t;
		}
		int length()
		{
			return l;
		}
		~ThreadList<T>()
		{
			while (front)
				pop_front();
		}
		ThreadListContainer_ptr getFront()
		{
			return front;
		}
		T remove(ThreadListContainer_ptr ptr)
		{
			if (ptr == NULL)
				return NULL;
			T t = NULL;
			//Search before
			ThreadListContainer_ptr before = NULL;
			if (ptr != front)
			{
				before = front;
				while (before)
				{
					if (before->next == ptr)
						break;
					before = before->next;
				}
				if (before == NULL)
					return NULL;
			}
			pthread_mutex_lock (&remove_mutex);
			if (before)
				before->next = ptr->next;
			else
				front = ptr->next;
			if (ptr == back)
				back = before;
			pthread_mutex_unlock (&remove_mutex);
			t = ptr->t;
			free(ptr);
			l--;
			return t;
		}
	private:
		volatile ThreadListContainer_ptr front;
		volatile ThreadListContainer_ptr back;
		pthread_mutex_t remove_mutex;
		int l;
};
