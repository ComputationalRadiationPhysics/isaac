'use strict'

function mulMatrix(matrix1, matrix2) {
	let result = new Array(16);
	for (let x = 0; x < 4; x++)
		for (let y = 0; y < 4; y++)
			result[y + x * 4] = matrix1[y + 0 * 4] * matrix2[0 + x * 4]
				+ matrix1[y + 1 * 4] * matrix2[1 + x * 4]
				+ matrix1[y + 2 * 4] * matrix2[2 + x * 4]
				+ matrix1[y + 3 * 4] * matrix2[3 + x * 4];
	return result;
}

function mulMatrix4(matrix, p) {
	let t = new Array(4);
	t[0] = matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12] * p[3];
	t[1] = matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13] * p[3];
	t[2] = matrix[2] * p[0] + matrix[6] * p[1] + matrix[10] * p[2] + matrix[14] * p[3];
	t[3] = matrix[3] * p[0] + matrix[7] * p[1] + matrix[11] * p[2] + matrix[15] * p[3];
	return t;
}

function rotateMatrix(x, y, z, rad) {
	//Rotation matrix:
	let s = Math.sin(rad * Math.PI / 180.0);
	let c = Math.cos(rad * Math.PI / 180.0);
	let l = Math.sqrt(x * x + y * y + z * z);
	if (l == 0)
		return;
	x = x / l;
	y = y / l;
	z = z / l;
	let rotate = new Array(9);
	rotate[0] = c + x * x * (1 - c);
	rotate[3] = x * y * (1 - c) - z * s;
	rotate[6] = x * z * (1 - c) + y * s;
	rotate[1] = y * x * (1 - c) + z * s;
	rotate[4] = c + y * y * (1 - c);
	rotate[7] = y * z * (1 - c) - x * s;
	rotate[2] = z * x * (1 - c) - y * s;
	rotate[5] = z * y * (1 - c) + x * s;
	rotate[8] = c + z * z * (1 - c);
	return rotate;
}

function mulMatrix3(matrix, p) {
	return mulMatrix4(matrix, [p[0], p[1], p[2], 1]);
}


class IsaacImageRenderer {

	/**
	 * 
	 * @param {Canvas} canvas 
	 */
	constructor(canvas) {
		this.translationMatrix = [];
		this.rotationMatrix = [];
		this.distanceMatrix = [];

		this.projectionMatrix = [
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		];

		this.modelViewMatrix = [];

		this.rotation = [0, 0, 0];
		this.distance = 0;
		this.position = [0, 0, 0];

		//set matrices
		this.setPosition([0, 0, 0]);
		this.setRotation([0, 0, 0]);
		this.setDistance(0);
		this.updateModelViewMatrix();

		this.canvas = canvas;
		this.ctx = this.canvas.getContext("2d");

		this.metadata = IsaacMetadata;

		this.wireframe = {
			enabled: true,
			color: "#ffffff"
		};

		this.useOneWindow = true;

		this.streamImage = new Image();

		this.size = {
			width: 0,
			height: 0,
			depth: 0
		};

		this.simulationBorderVertices = [];
	}

	setSimulationSize(width, height, depth) {
		this.size.width = width;
		this.size.height = height;
		this.size.depth = depth;
		let maxSize = Math.max(width, height, depth);

		this.simulationBorderVertices = [
			[-width / maxSize, height / maxSize, depth / maxSize],
			[width / maxSize, height / maxSize, depth / maxSize],
			[width / maxSize, -height / maxSize, depth / maxSize],
			[-width / maxSize, -height / maxSize, depth / maxSize],
			[-width / maxSize, height / maxSize, -depth / maxSize],
			[width / maxSize, height / maxSize, -depth / maxSize],
			[width / maxSize, -height / maxSize, -depth / maxSize],
			[-width / maxSize, -height / maxSize, -depth / maxSize]
		];
	}

	setProjectionMatrix(projectionMatrix) {
		this.projectionMatrix = projectionMatrix;
	}

	setPosition(position) {
		this.position = position;
		this.translationMatrix = [
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			position[0], position[1], position[2], 1
		];
	}

	getPosition() {
		return this.position;
	}

	setDistance(distance) {
		this.distance = distance;
		this.distanceMatrix = [
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, distance, 1
		];
	}

	getDistance() {
		return this.distance;
	}

	setRotation(rotation) {
		this.rotation = rotation;
		this.rotationMatrix = [
			rotation[0], rotation[1], rotation[2], 0,
			rotation[3], rotation[4], rotation[5], 0,
			rotation[6], rotation[7], rotation[8], 0,
			0, 0, 0, 1
		];
	}

	getRotation() {
		return this.rotation;
	}

	setWireframe(enabled, color) {
		this.wireframe.enabled = enabled;
		this.wireframe.color = color;
	}

	setOneWindow(useOneWindow) {
		this.useOneWindow = useOneWindow;
	}

	setStreamImage(streamImageSrc) {
		this.streamImage.src = streamImageSrc;
	}

	/**
	 * 
	 * @param {IsaacMetadata} metadata 
	 */
	setMetadata(metadata) {
		this.metadata = metadata;
	}

	updateModelViewMatrix() {
		this.modelViewMatrix = mulMatrix(this.distanceMatrix, mulMatrix(this.rotationMatrix, this.translationMatrix));
	}

	update() {
		this.updateModelViewMatrix();
	}

	draw() {
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		if (this.canvas.clientWidth != this.streamImage.width || this.canvas.clientHeight != this.streamImage.height) {
			this.ctx.beginPath();
			this.ctx.rect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
			this.ctx.fillStyle = "#000000";
			this.ctx.fill();
		}
		this.ctx.drawImage(this.streamImage, (parseInt(this.canvas.clientWidth) - parseInt(this.streamImage.width)) / 2, (parseInt(this.canvas.clientHeight) - parseInt(this.streamImage.height)) / 2);
		this.ctx.fillStyle = "#000000";
		if (this.useOneWindow) {
			this.ctx.globalAlpha = 0.5;
			this.ctx.strokeStyle = this.wireframe.color;
		}
		else {
			this.ctx.strokeStyle = '#000000';
		}
		this.ctx.lineWidth = 2;
		this.ctx.beginPath();

		if (this.wireframe.enabled) {
			this.drawSimulationBorders();
		}
		/*for (p in particles)
		{
			point( particles[p], false);
		}*/
		this.ctx.stroke();
		this.ctx.globalAlpha = 1.0;

		if (this.canvas.clientWidth == this.streamImage.width && this.canvas.clientHeight == this.streamImage.height) {
			this.ctx.font = "30px Arial";
			this.ctx.fillStyle = this.wireframe.color;
			this.ctx.textAlign = "right";
			this.ctx.fillText(parseInt(Math.round(this.metadata.particle_count / 1000000)) + " Mio. Particles", parseInt(this.canvas.clientWidth) - 10, parseInt(32));
			this.ctx.fillText(parseInt(Math.round(this.metadata.cell_count / 1000000)) + " Mio. Cells", parseInt(this.canvas.clientWidth) - 10, parseInt(57));
			this.ctx.fillText(parseInt(this.metadata.nodes) + " GPUs", parseInt(this.canvas.clientWidth) - 10, parseInt(this.canvas.clientHeight) - 10);
			this.ctx.textAlign = "left";
			this.ctx.fillText("Step " + parseInt(this.metadata.time_step), parseInt(10), parseInt(this.canvas.clientHeight) - 10);
			this.ctx.fillText(parseInt(Math.round(this.metadata.simulation_time / 1000)) + " ms Simulation", parseInt(10), parseInt(32));
			this.ctx.fillText(parseInt(Math.round(this.metadata.drawing_time / 1000)) + " ms Rendering", parseInt(10), parseInt(57));
		}
	}

	drawLine(pointFrom, pointTo, transform = true) {
		if (transform) {
			pointFrom = this.projectPoint(pointFrom);
			pointTo = this.projectPoint(pointTo);
		}

		if ((Math.abs(pointFrom[2]) > 1) || (Math.abs(pointTo[2]) > 1))
			return;
		this.ctx.moveTo(pointFrom[0], pointFrom[1]);
		this.ctx.lineTo(pointTo[0], pointTo[1]);
	}

	drawPoint(point, depthCheck = true, transform = true) {
		if (transform) {
			point = this.projectPoint(point);
		}
		if (!depthCheck && Math.abs(point[2]) > 1)
			return;
		this.ctx.fillRect(point[0], point[1], 3, 3);
	}

	drawSimulationBorders() {
		//front
		this.drawLine(this.simulationBorderVertices[0], this.simulationBorderVertices[1]);
		this.drawLine(this.simulationBorderVertices[1], this.simulationBorderVertices[2]);
		this.drawLine(this.simulationBorderVertices[2], this.simulationBorderVertices[3]);
		this.drawLine(this.simulationBorderVertices[3], this.simulationBorderVertices[0]);
		//back
		this.drawLine(this.simulationBorderVertices[4], this.simulationBorderVertices[5]);
		this.drawLine(this.simulationBorderVertices[5], this.simulationBorderVertices[6]);
		this.drawLine(this.simulationBorderVertices[6], this.simulationBorderVertices[7]);
		this.drawLine(this.simulationBorderVertices[7], this.simulationBorderVertices[4]);
		//connetion
		this.drawLine(this.simulationBorderVertices[0], this.simulationBorderVertices[4]);
		this.drawLine(this.simulationBorderVertices[1], this.simulationBorderVertices[5]);
		this.drawLine(this.simulationBorderVertices[2], this.simulationBorderVertices[6]);
		this.drawLine(this.simulationBorderVertices[3], this.simulationBorderVertices[7]);
	}

	projectPoint(point) {
		var maxSize = mulMatrix4(this.projectionMatrix, mulMatrix3(this.modelViewMatrix, point));
		var t = new Array(2);
		t[0] = this.canvas.clientWidth / 2 + maxSize[0] / maxSize[3] * this.canvas.clientWidth / 2;
		t[1] = this.canvas.clientHeight / 2 + maxSize[1] / maxSize[3] * this.canvas.clientHeight / 2;
		t[2] = maxSize[2] / maxSize[3];
		return t;
	}
}