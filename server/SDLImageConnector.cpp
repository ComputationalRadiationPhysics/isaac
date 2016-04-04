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

#include "SDLImageConnector.hpp"

SDLImageConnector::SDLImageConnector()
{
	SDL_Init(SDL_INIT_VIDEO);
	group = NULL;
	showClient = false;
}

std::string SDLImageConnector::getName()
{
	return "SDLImageConnector";
}

errorCode SDLImageConnector::init(int minport,int maxport)
{
	return 0;
}

errorCode SDLImageConnector::run()
{
	window = SDL_SetVideoMode( 512, 512, 32, SDL_HWSURFACE );
	int finish = 0;
	while (finish == 0)
	{
		SDL_Event event;
		while ( SDL_PollEvent( &event ) == 1 )
		{
			if (event.type == SDL_QUIT)
				finish = 2;
		}
		ImageBufferContainer* message;
		while (message = clientGetMessage())
		{
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			if (message->type == GROUP_FINISHED)
			{
				if (group == message->group)
				{
					group = NULL;
					SDL_FreeSurface(window);
					window = SDL_SetVideoMode( 512, 512, 32, SDL_HWSURFACE );
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				if (group == NULL)
				{
					group = message->group;
					SDL_FreeSurface(window);
					window = SDL_SetVideoMode( group->getFramebufferWidth(), group->getFramebufferHeight(), 32, SDL_HWSURFACE );
				}
				if (group == message->group) //We show always the very first group
				{
					SDL_LockSurface( window );
					uint8_t* pixels = (uint8_t*)(window->pixels);
					for (int i = 0; i < message->group->getVideoBufferSize() / 4; i++)
					{
						for (int j = 0; j < 3; j++)
							pixels[i*4+j] = message->image->buffer[i*4+2-j];
						pixels[i*4+3] = message->image->buffer[i*4+3];
					}
					SDL_UnlockSurface( window );
					SDL_Flip(window);
				}
			}
			clientSendMessage( message );
		}
		usleep(1000);
	}
	SDL_FreeSurface(window);
	SDL_Quit();
	//Even if the window is close we still need to free the video buffers!
	while (finish == 2)
	{
		ImageBufferContainer* message;
		while (message = clientGetMessage())
		{
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			clientSendMessage( message );
		}
		usleep(1000);
	}	
}
