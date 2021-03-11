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

#ifndef __IMAGECONNECTOR
#define __IMAGECONNECTOR


#include "Broker.hpp"
#include "MessageAble.hpp"
#include "Runable.hpp"
#include "ThreadList.hpp"
class Broker;

class ImageConnector
    : public Runable
    , public MessageAble<ImageBufferContainer>
{
public:
    ImageConnector();
    // To be overwritten
    virtual errorCode init(int minport, int maxport) = 0;
    virtual errorCode run() = 0;
    virtual std::string getName() = 0;

    // Called from the Master
    void setBroker(Broker* broker);

    bool showClient;

protected:
    Broker* broker;
};

#endif
