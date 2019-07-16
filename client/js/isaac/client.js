'use strict'

const IsaacResponseTypes = {
    hello: "hello",
    register: "register",
    period: "period",
    update: "update",
    exit: "exit"
};

const IsaacRequestTypes = {
    observe: "observe",
    stopObserve: "stop",
    exit: "exit"
}

const IsaacExitRequest = {
    type: IsaacRequestTypes.exit
};

const IsaacObserveRequest = {
    "type": IsaacRequestTypes.observe,
    "stream": 0,
    "dropable": undefined,
    "observer id": 0
};

const IsaacStopObserveRequest = {
    "type": IsaacRequestTypes.stopObserve,
    "observer id": 0
};

var IsaacResponse = {
    "type": "",
    "name": "",
    "id": -1,
    "nodes": 0,
    "max functors": 0,
    "functors": [],
    "sources": [],
    "metadata": [],
    "width": 0,
    "height": 0,
    "depth": 0,
    "dimension": 0,
    "projection": undefined,
    "position": undefined,
    "distance": undefined,
    "rotation": undefined,
    "interpolation": false,
    "step": 0.1,
    "iso surface": false,
    "ao isEnabled": false,
    "ao maxCellParticles": 1,
    "framebuffer width": 0,
    "framebuffer height": 0,
    "payload": {},

};

var IsaacFunctor = {
    "name": "",
    "description": ""
};

var IsaacSources = {
    "name": "",
    "feature dimension": 0
};

var IsaacMetadata = {
    "particle count": 0,
    "time step": 0,
    "cell count": 0,
    "simulation_time": 0,
    "drawing_time": 0,
    "particle_count": 0,
    "time_step": 0,
    "cell_count": 0,
    "simulation_time": 0,
    "drawing_time": 0
};

var IsaacFramebuffer = {
    width: 0,
    height: 0
};


class IsaacClient {
    constructor() {
        this.socket            = undefined;
        this.onOpenCallback    = undefined;
        this.onCloseCallback   = undefined;
        this.onMessageCallback = undefined;
        this.onErrorCallback   = undefined;
        this.observerId = -1;
    }


    connect(url, port) {
        //connecto to websocket server
        url = "ws://" + url + ":" + port;
        let protocol = "isaac-json-protocol";
        this.socket = new WebSocket(url, protocol);

        //set websocket callbacks to class functions
        //set callbacks for outside use
        this.socket.onopen = (function(e) {
            this.onOpen(e);
        }).bind(this);

        this.socket.onmessage = (function(e) {
            this.onMessage(e);
        }).bind(this);

        this.socket.onerror = (function(e) {
            this.onError(e);
        }).bind(this);

        this.socket.onerror = (function(e) {
            this.onClose(e);
        }).bind(this);
    }

    /**
     * sends IsaacExitRequest and closes websocket
     * 
     */
    close() {
        this.socket.send(JSON.stringify(IsaacExitRequest));
        this.socket.close();
        this.onClose();
    }

    send(msg) {
        this.socket.send(JSON.stringify(msg));
    }

    /**
     * send a observe request to server
     * @param {Number} stream_id 
     * @param {Boolean} dropable 
     * @param {Number} observe_id 
     */
    requestObserve(stream_id, dropable, observe_id) {
        let request = IsaacObserveRequest;
        request.stream = stream_id;
        request.dropable = dropable;
        request["observe id"] = observe_id;
        this.socket.send(JSON.stringify(request));
        this.observerId = observe_id;
    }

    /**
     * send a stop observe request
     * @param {*} observe_id 
     */
    requestStopObserve(observe_id) {
        let request = IsaacStopObserveRequest;
        request["observer id"] = observe_id;
        this.socket.send(JSON.stringify(request));
        this.observerId = -1;
    }

    getOberserID() {
        return this.observerId;
    }

    /**
     * returns true if websocket is connected
     */
    isConnected() {
        return this.socket && this.socket.readyState == 1;
    }

    //websocket callbacks 
    //these function will preprocess the input messages

    onOpen(e) {
        if(this.onOpenCallback) {
            this.onOpenCallback(e);
        }
    }

    onMessage(e) {
        if(this.onMessageCallback) {
            let obj = JSON.parse(e.data);
            this.onMessageCallback(obj);
        }
    }

    onError(e) {
        if(this.onErrorCallback) {
            this.onErrorCallback(e);
        }
    }

    onClose(e) {
        if(this.onCloseCallback) {
            this.onCloseCallback(e);
        }
    }


    //setter for websocket internal callbacks

    setOnOpen(callback) {
        this.onOpenCallback = callback;
    }

    setOnMessage(callback) {
        this.onMessageCallback = callback;
    }

    setOnError(callback) {
        this.onErrorCallback = callback;
    }

    setOnClose(callback) {
        this.onCloseCallback = callback;
    }
};