/**
 * ISAAC Client - Reference client implementation for ISAAC
 *
 * @link    https://github.com/ComputationalRadiationPhysics/isaac
 * @license GPLv3
 * @author  Alexander Matthes, Vincent Gerber
 */
var video = document.getElementById("video");
var preview = document.getElementById("preview");
var p_ctx = preview.getContext("2d");
var transfer = document.getElementById("transfer");
var t_ctx = transfer.getContext("2d");

var observe_id = -1;
var particles = null;
var plugins = new Array();
var dropable = document.getElementById("dropable_checkbox").checked;
document.getElementById("preview_div").style.opacity = '1.0';
const default_image_src = "data:image/gif;base64,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";


var functions = new Array();
var weight = new Array();


var transfer_array = [];
var transfer_points = [];

p_ctx.font = "10px Arial";
//redraw(preview, position, rotation, distance, stream_img, particles, wireframe, one_window, lastIsaacState.metadata);


preview.addEventListener("mousewheel", mouseWheelHandler, false);
preview.addEventListener("DOMMouseScroll", mouseWheelHandler, false);
preview.addEventListener("mousemove", mouseMoveHandler, false);
preview.addEventListener("mousedown", mouseDownHandler, false);
preview.addEventListener("mouseup", mouseUpHandler, false);
preview.onmouseout = function () { mouseResetHandler(); document.body.style.overflowY = "scroll"; };

transfer.addEventListener("mousedown", transferMouseDownHandler, false);

document.getElementById("video_td").style.display = 'none';
document.getElementById("preview_td").style.display = 'none';




let renderer = new IsaacImageRenderer(preview);
let client = new IsaacClient();
let lastIsaacState = IsaacResponse;


var last_feedback_send = 0;
var sources = null;


/**
 * Cycle for drawing
 */
onRender = function(timestep) {
	if(client.getOberserID() != -1) {
		renderer.draw(); 
	}
	requestAnimationFrame(onRender);
}

requestAnimationFrame(onRender);


onload = function () {
	renderer.setStreamImage(default_image_src);
	renderer.setWireframe(document.getElementById("wireframe_checkbox").checked, "#ffffff");
	renderer.setOneWindow(document.getElementById("one_window_checkbox").checked);

	client.setOnOpen(onClientOpen);
	client.setOnMessage(onClientMessage);
	client.setOnError(onClientError);
	client.setOnClose(onClientClosed);
}


function sendFeedback(name, feedback) {
	var d = new Date();
	last_feedback_send = d.getTime();
	var message =
	{
		"type": "feedback",
		"observe id": observe_id
	};
	message[name] = feedback;
	client.send(message);

}

function zoom(delta) {
	renderer.setDistance(renderer.getDistance() + delta);
	renderer.update();

	p_ctx.fillText("Delta: " + delta, 1, 10);
	sendFeedback("distance relative", delta);
}

function mouseWheelHandler(e) {
	document.body.style.overflowY = "hidden";
	zoom(Math.max(-1, Math.min(1, (e.wheelDelta || -e.detail))) * 0.1);
}

var mousedown = [0, 0, 0];
var mousex = 0;
var mousey = 0;
function mouseDownHandler(e) {
	mousedown[e.button] = 1;
	mousex = e.clientX;
	mousey = e.clientY;
}
function mouseUpHandler(e) {
	mousedown[e.button] = 0;
}
function mouseResetHandler() {
	mousedown[0] = 0;
	mousedown[1] = 0;
	mousedown[2] = 0;
}

function move(add) {
	let rotation = renderer.getRotation();
	let position = renderer.getPosition();
	var add_p = [
		rotation[0] * add[0] + rotation[1] * add[1] + rotation[2] * add[2],
		rotation[3] * add[0] + rotation[4] * add[1] + rotation[5] * add[2],
		rotation[6] * add[0] + rotation[7] * add[1] + rotation[8] * add[2]
	];
	position[0] += add_p[0];
	position[1] += add_p[1];
	position[2] += add_p[2];

	sendFeedback("position relative", add);

	renderer.setPosition(position);
	renderer.update();
}

function rotate(dx, dy, dz) {
	var l = Math.sqrt(dx * dx + dy * dy + dz * dz);
	if (l > 0) {
		dx /= l;
		dy /= l;
		dz /= l;
		let add = rotateMatrix(dy, dx, dz, l / 2);
		let rotation = renderer.getRotation();
		let result = new Array(9);
		for (x = 0; x < 3; x++)
			for (y = 0; y < 3; y++)
				result[y + x * 3] = add[y + 0 * 3] * rotation[0 + x * 3]
					+ add[y + 1 * 3] * rotation[1 + x * 3]
					+ add[y + 2 * 3] * rotation[2 + x * 3];
		rotation = result;
		renderer.setRotation(rotation);
		renderer.update();

		sendFeedback("rotation axis", [dy, dx, dz, l / 2]);
	}
}

function mouseMoveHandler(e) {
	var dx = e.clientX - mousex;
	var dy = e.clientY - mousey;
	if (mousedown[0]) {
		rotate(dx, -dy, 0);
	}
	if (mousedown[2] || mousedown[1]) {
		move([dx / 100, dy / 100, 0]);
	}
	mousex = e.clientX;
	mousey = e.clientY;
}



function stopObserverSimulation() {
	if (observe_id < 0)
		return;
	
	client.requestStopObserve(observe_id);
	observe_id = -1;
	var transfer_form = document.getElementById("transfer_form");
	while (transfer_form.firstChild) {
		transfer_form.removeChild(transfer_form.firstChild);
	}
	renderer.setStreamImage(default_image_src);
}

function observeSimulation(id) {
	stopObserverSimulation();
	observe_id = id;
	var connector_id = 2; //jpeg uri
	client.requestObserve(connector_id, dropable, observe_id);
}

function transferRectangle(x, y, w, h, color) {
	t_ctx.beginPath();
	t_ctx.rect(x, y, w, h);
	t_ctx.fillStyle = color;
	t_ctx.fill();
}

var transfer_high = 64;

function transferMouseDownHandler(e) {
	var rect = transfer.getBoundingClientRect();
	var X = Math.floor(e.clientX - rect.left);
	var Y = Math.floor(e.clientY - rect.top);
	var source = Math.floor(Y / (transfer_high + 2));
	var value = Math.round(X / transfer.width * transfer_array[source].length);
	var localY = Math.floor(Y - source * (transfer_high + 2));
	var element =
	{
		"value": value,
		"r": transfer_array[source][value][0] / 255,
		"g": transfer_array[source][value][1] / 255,
		"b": transfer_array[source][value][2] / 255,
		"a": 1 - localY / (transfer_high + 2)
	}
	transfer_points[source].push(element);
	sendFeedback("transfer points", transfer_points);
}

function updateTransfer() {
	transfer.width = video.width;
	transfer.height = transfer_array.length * transfer_high + (transfer_array.length - 1) * 2;
	for (var f = 0; f < transfer_array.length; f++) {
		var lastX = 0;
		for (var i = 0; i < transfer_array[f].length; i++) {
			var newX = Math.round(i * transfer.width / transfer_array[f].length);
			if (newX != lastX) {
				var bg1 = [127, 127, 127];
				var bg2 = [255, 255, 255];
				if (Math.round((lastX + transfer_high / 4) / (transfer_high / 2)) % 2 == 0) {
					var bgt = bg1;
					bg1 = bg2;
					bg2 = bgt;
				}
				var rgba = transfer_array[f][i];
				var fg1 = new Array(3);
				fg1[0] = Math.round((rgba[0] * rgba[3] + bg1[0] * (255 - rgba[3])) / 255);
				fg1[1] = Math.round((rgba[1] * rgba[3] + bg1[1] * (255 - rgba[3])) / 255);
				fg1[2] = Math.round((rgba[2] * rgba[3] + bg1[2] * (255 - rgba[3])) / 255);
				var fg2 = new Array(3);
				fg2[0] = Math.round((rgba[0] * rgba[3] + bg2[0] * (255 - rgba[3])) / 255);
				fg2[1] = Math.round((rgba[1] * rgba[3] + bg2[1] * (255 - rgba[3])) / 255);
				fg2[2] = Math.round((rgba[2] * rgba[3] + bg2[2] * (255 - rgba[3])) / 255);
				transferRectangle(lastX, f * (transfer_high + 2), newX - lastX, transfer_high / 2, "rgb(" + fg1[0] + "," + fg1[1] + "," + fg1[2] + ")");
				transferRectangle(lastX, f * (transfer_high + 2) + transfer_high / 2, newX - lastX, transfer_high / 2, "rgb(" + fg2[0] + "," + fg2[1] + "," + fg2[2] + ")");
			}
			lastX = newX;
		}
		t_ctx.beginPath();
		var first = true;
		for (p in transfer_points[f]) {
			var point = [
				transfer_points[f][p].value * transfer.width / transfer_array[f].length,
				f * (transfer_high + 2) + transfer_high - transfer_points[f][p].a * transfer_high
			];
			if (first) {
				t_ctx.moveTo(point[0], point[1]);
				first = false;
			}
			else
				t_ctx.lineTo(point[0], point[1]);
		}
		t_ctx.stroke();
		transferRectangle(0, f * (transfer_high + 2) + transfer_high, transfer.width, 2, "black");
	}
}

function removePoint(e) {
	var elements = e.target.id.split("_");
	var s = parseInt(elements[1]);
	var i = parseInt(elements[2]);
	transfer_points[s].splice(i, 1);
	sendFeedback("transfer points", transfer_points);
}

function pad(num, size) {
	var s = num.toString();
	var l = size - s.length;
	for (var i = 0; i < l; i++)
		s = "&nbsp;" + s;
	return s;
}

document.getElementById("interval_range").oninput = function () {
	var interval_range = document.getElementById("interval_range");
	var interval_span = document.getElementById("interval_span");
	interval_span.innerHTML = interval_range.value;
};

document.getElementById("interval_change").onclick = function () {
	var metadata =
	{
		"interval": parseInt(document.getElementById("interval_range").value)
	};
	sendFeedback("metadata", metadata);
};

document.getElementById("clip_add").onclick = function () {
	var clipping_add =
	{
		"position": [parseFloat(document.getElementById("clip_px").value), parseFloat(document.getElementById("clip_py").value), parseFloat(document.getElementById("clip_pz").value)],
		"normal": [parseFloat(document.getElementById("clip_nx").value), parseFloat(document.getElementById("clip_ny").value), parseFloat(document.getElementById("clip_nz").value)]
	};
	sendFeedback("clipping add", clipping_add);
};

document.getElementById("clip_bx").onclick = function () {
	document.getElementById("clip_nx").value = "1";
	document.getElementById("clip_ny").value = "0";
	document.getElementById("clip_nz").value = "0";
};

document.getElementById("clip_by").onclick = function () {
	document.getElementById("clip_nx").value = "0";
	document.getElementById("clip_ny").value = "1";
	document.getElementById("clip_nz").value = "0";
};

document.getElementById("clip_bz").onclick = function () {
	document.getElementById("clip_nx").value = "0";
	document.getElementById("clip_ny").value = "0";
	document.getElementById("clip_nz").value = "1";
};

function createRemoveClippingFeedback(c) {
	return function () {
		sendFeedback("clipping remove", parseInt(c));
	};
}

function createEditClippingFeedback(c) {
	return function () {
		var clipping_edit =
		{
			"nr": parseInt(c),
			"position": [parseFloat(document.getElementById("clip_" + c + "_px").value), parseFloat(document.getElementById("clip_" + c + "_py").value), parseFloat(document.getElementById("clip_" + c + "_pz").value)],
			"normal": [parseFloat(document.getElementById("clip_" + c + "_nx").value), parseFloat(document.getElementById("clip_" + c + "_ny").value), parseFloat(document.getElementById("clip_" + c + "_nz").value)]
		};
		sendFeedback("clipping edit", clipping_edit);
	};
};

function createWeightFeedback(w, range_name, value_name) {
	return function () {
		var myself = document.getElementById(range_name);
		weight[w] = (myself.value / 100) * (myself.value / 100);
		var value = document.getElementById(value_name);
		if (weight[w] == 0)
			value.innerHTML = ' <b><font color="red">Off</font></b> ';
		else
			value.innerHTML = ' <b><font color="green">' + weight[w].toFixed(1) + '</font></b> ';
	};
}

function createOptimizedFunction(i, minmax, with_add) {
	return function () {
		var mul = 1 / (minmax["max"] - minmax["min"]);
		var function_string = document.getElementById("f_" + i + "_string");
		if (function_string.value == "idem" || function_string.value == "")
			functions[i] = "mul(" + mul.toFixed(5) + ")";
		else
			functions[i] = function_string.value + " | mul(" + mul.toFixed(5) + ")";
		if (with_add)
			functions[i] += " | add(0.5)";
		function_string.value = functions[i];
	};
}

function updateMetaDataTable(metadata) {
	if (Object.keys(metadata).length === 0 && JSON.stringify(metadata) === JSON.stringify({}))
		return;
	for (element in metadata) {
		if (element == "particle count")
			lastIsaacState.metadata.particle_count = metadata[element];
		if (element == "time step")
			lastIsaacState.metadata.time_step = metadata[element];
		if (element == "cell count")
			lastIsaacState.metadata.cell_count = metadata[element];
		if (element == "simulation_time")
			lastIsaacState.metadata.simulation_time = metadata[element];
		if (element == "drawing_time")
			lastIsaacState.metadata.drawing_time = metadata[element];
	}
	renderer.setMetadata(lastIsaacState.metadata);
}

function RGB2HTML(red, green, blue) {
	var decColor = 0x1000000 + blue + 0x100 * green + 0x10000 * red;
	return '#' + decColor.toString(16).substr(1);
}

document.getElementById("connect").onclick = function () {
	if (!client.isConnected()) {
		let url = document.getElementById("server_url").value;
		let port = document.getElementById("server_port").value;
		client.connect(url, port);
		console.log("Connect to " + url + ":" + port);
	}
	else {
		client.close();
	}
};

function onClientOpen() {
	document.getElementById("status").textContent = "Status: Connected";
	document.getElementById("connect").value = "Close";
	console.log("Connected..");
}

/**
 * 
 * @param {IsaacResponse} response 
 */
function onClientMessage(response) {
	if (response["type"] == IsaacResponseTypes.hello) {
		document.getElementById("isaac_name").textContent = response["name"];
		var table = document.getElementById("list_table");
		while (table.rows.length > 1)
			table.deleteRow(1);
		document.getElementById("list_td").style.display = 'initial';
	}
	if (response["type"] == IsaacResponseTypes.register) {
		plugins.push(response);
		var table = document.getElementById("list_table");
		var row = table.insertRow(-1);
		row.id = response["id"];
		row.insertCell(0).innerHTML = response["name"];
		row.insertCell(1).innerHTML = response["id"];
		row.insertCell(2).innerHTML = response["nodes"];
		lastIsaacState.metadata.nodes = response["nodes"];
		row.insertCell(3).innerHTML = response["max functors"];

		var cell = row.insertCell(4);
		for (element in response["functors"]) {
			var e = response["functors"][element]
			cell.innerHTML += "<b>" + e["name"] + ":</b> ";
			cell.innerHTML += e["description"] + "<br/>";
		}
		var cell = row.insertCell(5);
		var d = response["dimension"];
		cell.innerHTML += response["width"];
		if (d > 1)
			cell.innerHTML += " * " + response["height"];
		if (d > 2)
			cell.innerHTML += " * " + response["depth"];
		var cell = row.insertCell(6);
		for (element in response["sources"]) {
			var e = response["sources"][element]
			cell.innerHTML += "<b>" + e["name"] + ":</b><br/>";
			cell.innerHTML += " Feature dimension(" + e["feature dimension"] + ")<br/>";
		}
		var cell = row.insertCell(7);//.innerHTML = response["metadata"].toSource();
		for (element in response["metadata"]) {
			cell.innerHTML += "<b>" + element + ":</b> ";
			cell.innerHTML += response["metadata"][element];
			cell.innerHTML += "<br/>";
		}
		var observeCell = row.insertCell(8);
		var button = document.createElement("input");
		button.type = "button";
		button.value = "Observe";
		button.onclick = function () {
			renderer.setSimulationSize(response["width"], response["height"], response["depth"])

			document.getElementById("video_td").style.display = 'table-cell';
			document.getElementById("preview_td").style.display = 'table-cell';

			renderer.setRotation(response["rotation"]);
			renderer.setPosition(response["position"]);
			renderer.setDistance(response["distance"]);
			renderer.setProjectionMatrix(response["projection"]);
			renderer.update();

			sources = response["sources"];
			document.getElementById("interpolation_checkbox").checked = response["interpolation"];
			document.getElementById("step").value = response["step"];
			document.getElementById("iso_surface_checkbox").checked = response["iso surface"];
			
			aoSetValues(response);

			preview.width = response["framebuffer width"];
			preview.height = response["framebuffer height"];
			video.width = response["framebuffer width"];
			video.height = response["framebuffer height"];
			observeSimulation(response["id"]);
		};
		observeCell.appendChild(button);
	}
	if (response["type"] == IsaacResponseTypes.period || response["type"] == IsaacResponseTypes.update) {
		if (response.hasOwnProperty("payload")) {
			renderer.setStreamImage(response["payload"]);
		}
		if (response.hasOwnProperty("metadata")) {
			if (response["metadata"].hasOwnProperty("interval")) {
				var interval_range = document.getElementById("interval_range");
				var interval_span = document.getElementById("interval_span");
				interval_range.value = response["metadata"]["interval"];
				interval_span.innerHTML = response["metadata"]["interval"];
				delete response["metadata"]["interval"];
			}
			updateMetaDataTable(response["metadata"]);
			if (response["metadata"].hasOwnProperty("reference particles"))
				particles = response["metadata"]["reference particles"];
		}
		var d = new Date();
		var now = d.getTime();
		if (now - last_feedback_send > 100) {
			if (response.hasOwnProperty("projection"))
				renderer.setProjectionMatrix(response["projection"]);
			if (response.hasOwnProperty("position"))
				renderer.setPosition(response["position"]);
			if (response.hasOwnProperty("rotation"))
				renderer.setRotation(response["rotation"]);
			if (response.hasOwnProperty("distance"))
				renderer.setDistance(response["distance"]);
			if (response.hasOwnProperty("interpolation"))
				document.getElementById("interpolation_checkbox").checked = response["interpolation"];
			if (response.hasOwnProperty("step"))
				document.getElementById("step").checked = response["step"];
			if (response.hasOwnProperty("iso surface"))
				document.getElementById("iso_surface_checkbox").checked = response["iso surface"];

			aoSetValues(response);

			renderer.update();
		}


		processFunctions(response);

		processEyeDistance(response);

		processMinMax(response);

		processWeight(response);

		processTransferArray(response);

		processClipping(response);

		processBackground(response);
	}
	if (response["type"] == IsaacResponseTypes.exit) {
		var table = document.getElementById("list_table");
		for (var c = 1; c < table.rows.length; c++)
			if (table.rows[c].id == response["id"]) {
				table.deleteRow(c);
				break;
			}
		if (observe_id == response["id"]) {
			document.getElementById("video_td").style.display = 'none';
			document.getElementById("preview_td").style.display = 'none';
		}
		renderer.setStreamImage(default_image_src);
	}
}

function onClientError(e) {

}

function onClientClosed(e) {
	if (document.getElementById("status").textContent != "Status: An error occured. Wrong address, network gone or server not started?")
		document.getElementById("status").textContent = "Status: Closed";
	document.getElementById("connect").value = "Connect";
	document.getElementById("list_td").style.display = 'none';
	document.getElementById("video_td").style.display = 'none';
	document.getElementById("preview_td").style.display = 'none';
}

function processFunctions(response) {
	if (response.hasOwnProperty("functions")) {
		var functions_div = document.getElementById("functions_div");
		while (functions_div.firstChild) {
			functions_div.removeChild(functions_div.firstChild);
		}
		functions = response["functions"];
		for (var f in functions) {
			var title = document.createElement("SPAN");
			title.innerHTML = sources[f].name + " function: ";
			functions_div.appendChild(title);
			var function_string = document.createElement("INPUT");
			function_string.value = functions[f].source;
			function_string.size = 80;
			function_string.id = "f_" + f + "_string";
			functions_div.appendChild(function_string);
			var error = document.createElement("SPAN");
			switch (functions[f].error) {
				case 0:
					error.innerHTML = ' <b><font color="green">Ok</font></b>';
					break;
				case 1:
					error.innerHTML = ' <b><font color="orange">Too much functors, ignoring at least one</font></b>';
					break;
				case 2:
					error.innerHTML = ' <b><font color="orange">Too much parameters, ignoring at least one</font></b>';
					break;
				case -1:
					error.innerHTML = ' <b><font color="red">Missing )</font></b>';
					break;
				case -2:
					error.innerHTML = ' <b><font color="red">Unknown functor</font></b>';
					break;
				default:
					error.innerHTML = ' <b><font color="red">Fatal error</font></b>';
			}
			functions_div.appendChild(error);
			var minmax = document.createElement("SPAN");
			minmax.id = "i_" + f + "_string";
			functions_div.appendChild(minmax);
			functions_div.appendChild(document.createElement("BR"));
		}
		var function_button = document.createElement("INPUT");
		function_button.type = "button";
		function_button.value = "Update functions";
		function_button.id = "f_button";
		function_button.onclick = function () {
			for (var f in functions)
				functions[f] = document.getElementById("f_" + f + "_string").value;
			sendFeedback("functions", functions);
		};
		functions_div.appendChild(function_button);
		var minmax_button = document.createElement("INPUT");
		minmax_button.type = "button";
		minmax_button.value = "Update minmax interval";
		minmax_button.id = "i_button";
		minmax_button.onclick = function () {
			sendFeedback("request", "minmax");
		};
		functions_div.appendChild(minmax_button);
	}
}

function processEyeDistance(response) {
	if (response.hasOwnProperty("eye distance")) {
		var stereo_div = document.getElementById("stereo_div");
		while (stereo_div.firstChild) {
			stereo_div.removeChild(stereo_div.firstChild);
		}
		eye_distance = response["eye distance"];
		var title = document.createElement("SPAN");
		title.innerHTML = "Eye Distance: ";
		stereo_div.appendChild(title);
		var stereo_range = document.createElement("INPUT");
		stereo_range.type = "range";
		stereo_range.min = 0;
		stereo_range.max = 200;
		stereo_range.id = "stereo_range";
		stereo_range.value = Math.floor(eye_distance * 1000.0);
		stereo_div.appendChild(stereo_range);
		var submit = document.createElement("INPUT");
		submit.type = "button";
		submit.value = "Set distance";
		submit.onclick = function () {
			sendFeedback("eye distance", parseFloat(document.getElementById("stereo_range").value) / 1000.0);
		};
		stereo_div.appendChild(submit);
	}
}

function processMinMax(response) {
	if (response.hasOwnProperty("minmax")) {
		for (var i in response["minmax"]) {
			var parent = document.getElementById("i_" + i + "_string");
			while (parent.firstChild) {
				parent.removeChild(parent.firstChild);
			}
			var title = document.createElement("SPAN");
			title.innerHTML = " [ " + response["minmax"][i]["min"].toFixed(3) + " , " + response["minmax"][i]["max"].toFixed(3) + " ]";
			parent.appendChild(title);
			var apply = document.createElement("INPUT");
			apply.type = "button";
			apply.value = "Optimize";
			apply.id = "i_" + i + "_button";
			apply.onclick = createOptimizedFunction(i, response["minmax"][i], false);
			parent.appendChild(apply);
			apply = document.createElement("INPUT");
			apply.type = "button";
			apply.value = "With add";
			apply.id = "i_" + i + "_button_add";
			apply.onclick = createOptimizedFunction(i, response["minmax"][i], true);
			parent.appendChild(apply);
		}
	}
}

function processWeight(response) {
	if (response.hasOwnProperty("weight")) {
		var weight_div = document.getElementById("weight_div");
		while (weight_div.firstChild) {
			weight_div.removeChild(weight_div.firstChild);
		}
		weight = response["weight"];
		for (var w in weight) {
			var title = document.createElement("SPAN");
			title.innerHTML = sources[w].name + ": ";
			weight_div.appendChild(title);
			var weight_range = document.createElement("INPUT");
			weight_range.type = "range";
			weight_range.min = 0;
			weight_range.max = 447;
			weight_range.value = Math.floor(Math.sqrt(weight[w]) * 100);
			var range_name = "w_" + w + "_range";
			var value_name = "w_" + w + "_value";
			weight_range.id = range_name
			weight_range.oninput = createWeightFeedback(w, range_name, value_name);
			weight_div.appendChild(weight_range);
			var value = document.createElement("SPAN");
			if (weight[w] == 0)
				value.innerHTML = ' <b><font color="red">Off</font></b> ';
			else
				value.innerHTML = ' <b><font color="green">' + weight[w].toFixed(1) + '</font></b> ';
			value.id = value_name;
			weight_div.appendChild(value);
		}
		var weight_button = document.createElement("INPUT");
		weight_button.type = "button";
		weight_button.value = "Update weights";
		weight_button.id = "w_button";
		weight_button.onclick = function () {
			sendFeedback("weight", weight);
		};
		weight_div.appendChild(weight_button);
	}
}

function processTransferArray(response) {
	if (response.hasOwnProperty("transfer array")) {
		transfer_array = response["transfer array"];
		if (response.hasOwnProperty("transfer points")) {
			transfer_points = response["transfer points"];
			var transfer_form = document.getElementById("transfer_form");
			while (transfer_form.firstChild) {
				transfer_form.removeChild(transfer_form.firstChild);
			}
			for (var s in transfer_points) {
				var node = document.createElement("TD");
				node.innerHTML = "<b>" + sources[s].name + "</b>";
				node.style.vertical_align = "bottom";
				for (var i in transfer_points[s]) {
					var subnode = document.createElement("DIV");
					var transfer = document.createElement("INPUT");
					transfer.type = "text";
					transfer.value = (transfer_points[s][i].value / transfer_array[s].length).toFixed(3);
					transfer.size = 4;
					transfer.id = "p_" + s + "_" + i + "_transfer";
					if (i == 0 || i == transfer_points[s].length - 1)
						transfer.disabled = true;
					subnode.appendChild(transfer);
					var input = document.createElement("INPUT");
					input.type = "button";
					var picker = new jscolor(input);
					input.jscolor = picker;
					input.size = 6;
					picker.closable = true;
					picker.closeText = "Close";
					picker.fromRGB(transfer_points[s][i].r * 255, transfer_points[s][i].g * 255, transfer_points[s][i].b * 255);
					input.id = "p_" + s + "_" + i + "_color";
					subnode.appendChild(input);
					var alpha = document.createElement("INPUT");
					alpha.value = transfer_points[s][i].a;
					alpha.size = 10;
					alpha.id = "p_" + s + "_" + i + "_alpha";
					subnode.appendChild(alpha);
					if (i != 0 && i != transfer_points[s].length - 1) {
						var remove = document.createElement("INPUT");
						remove.type = "button";
						remove.value = "Remove";
						remove.id = "p_" + s + "_" + i + "_remove";
						remove.onclick = removePoint;
						subnode.appendChild(remove);
					}
					node.appendChild(subnode);
				}
				transfer_form.appendChild(node);
			}
			var node = document.createElement("TD");
			node.innerHTML = "<b> Controls </b><br/>";
			var redo = document.createElement("INPUT");
			redo.type = "button";
			redo.value = "Reset";
			redo.onclick = function () {
				sendFeedback("request", "transfer");
			};
			node.appendChild(redo);
			var submit = document.createElement("INPUT");
			submit.type = "button";
			submit.value = "Submit changes";
			submit.onclick = function () {
				for (var s in transfer_points)
					for (var i in transfer_points[s]) {
						var base = "p_" + s + "_" + i;
						var color = document.getElementById(base + "_color");
						var alpha = document.getElementById(base + "_alpha");
						transfer_points[s][i].r = color.jscolor.rgb[0] / 255;
						transfer_points[s][i].g = color.jscolor.rgb[1] / 255;
						transfer_points[s][i].b = color.jscolor.rgb[2] / 255;
						transfer_points[s][i].a = parseFloat(alpha.value);
						if (i > 0 && i < transfer_points[s].length - 1) {
							var transfer = document.getElementById(base + "_transfer");
							var transfer_value = Math.round(parseFloat(transfer.value) * transfer_array[s].length);
							if (transfer_value < 1)
								transfer_value = 1;
							if (transfer_value >= transfer_array[s].length)
								transfer_value = transfer_array[s].length - 1;
							transfer_points[s][i].value = transfer_value;
						}
					}
				sendFeedback("transfer points", transfer_points);
			};
			node.appendChild(submit);
			transfer_form.appendChild(node);
		}
		updateTransfer();
	}
}

function processClipping(response) {

	if (response.hasOwnProperty("clipping")) {
		clipping = response["clipping"];
		var clipping_div = document.getElementById("clipping_list_div");
		while (clipping_div.firstChild) {
			clipping_div.removeChild(clipping_div.firstChild);
		}
		for (var c in clipping) {
			var node = document.createElement("DIV");

			var span = document.createElement("SPAN");
			span.innerHTML = " Position: ";
			node.appendChild(span);

			var input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].position[0];
			input.size = 4;
			input.id = "clip_" + c + "_px";
			node.appendChild(input);

			input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].position[1];
			input.size = 4;
			input.id = "clip_" + c + "_py";
			node.appendChild(input);

			input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].position[2];
			input.size = 4;
			input.id = "clip_" + c + "_pz";
			node.appendChild(input);

			span = document.createElement("SPAN");
			span.innerHTML = " Normal: ";
			node.appendChild(span);

			input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].normal[0];
			input.size = 4;
			input.id = "clip_" + c + "_nx";
			node.appendChild(input);

			input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].normal[1];
			input.size = 4;
			input.id = "clip_" + c + "_ny";
			node.appendChild(input);

			input = document.createElement("INPUT");
			input.type = "text";
			input.value = clipping[c].normal[2];
			input.size = 4;
			input.id = "clip_" + c + "_nz";
			node.appendChild(input);

			span = document.createElement("SPAN");
			span.innerHTML = " ";
			node.appendChild(span);

			input = document.createElement("INPUT");
			input.type = "button";
			input.value = "Edit";
			input.onclick = createEditClippingFeedback(c);
			node.appendChild(input);

			span = document.createElement("SPAN");
			span.innerHTML = " ";
			node.appendChild(span);

			input = document.createElement("INPUT");
			input.type = "button";
			input.value = "Remove";
			input.onclick = createRemoveClippingFeedback(c);
			node.appendChild(input);

			clipping_div.appendChild(node);
		}
	}
}

function processBackground(response) {
	if (response.hasOwnProperty("background color")) {
		var background_color = document.getElementById("background_color");
		background_color.jscolor.fromRGB(
			response["background color"][0] * 255,
			response["background color"][1] * 255,
			response["background color"][2] * 255);
		updateWireframe();
	}
}

document.getElementById("stop").onclick = function () {
	document.getElementById("video_td").style.display = 'none';
	document.getElementById("preview_td").style.display = 'none';
	stopObserverSimulation();
}

document.getElementById("pause").onclick = function () {

	var metadata =
	{
		"pause": true
	};
	sendFeedback("metadata", metadata);
}

document.getElementById("exit").onclick = function () {
	if (confirm("Do you really want to stop the simulation?")) {
		var metadata =
		{
			"exit": 1
		};
		sendFeedback("metadata", metadata);
		document.getElementById("stop").onclick();
	}
}

document.getElementById("panic").onclick = function () {
	if (confirm("Do you really want to reset the view?")) {
		position = [0, 0, 0];
		distance = -4.5;
		rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
		sendFeedback("position absolute", position);
		sendFeedback("distance absolute", distance);
		sendFeedback("rotation absolute", rotation);
	}
}

document.getElementById("bounding").onclick = function () {
	sendFeedback("bounding box", true);
}

document.getElementById("background_change").onclick = function () {
	var background_color = new Array(3);
	background_color[0] = document.getElementById("background_color").jscolor.rgb[0] / 255;
	background_color[1] = document.getElementById("background_color").jscolor.rgb[1] / 255;
	background_color[2] = document.getElementById("background_color").jscolor.rgb[2] / 255;
	sendFeedback("background color", background_color);
}

function one_window_checkbox_click() {
	one_window = document.getElementById("one_window_checkbox").checked;
	renderer.setOneWindow(one_window);
	if (one_window) {
		var node = document.getElementById("preview_div");
		document.getElementById("video_container_div").appendChild(node);
	}
	else {
		var node = document.getElementById("preview_div");
		document.getElementById("preview_container_div").appendChild(node);
	}
};

function wireframe_checkbox_click() {
	updateWireframe();
};


function dropable_checkbox_click() {
	dropable = document.getElementById("dropable_checkbox").checked;
};

function interpolation_checkbox_click() {
	var status = document.getElementById("interpolation_checkbox").checked;
	sendFeedback("interpolation", status);
};

function step_button_click() {
	var status = parseFloat(document.getElementById("step").value);
	sendFeedback("step", status);
};

function iso_surface_checkbox_click() {
	var status = document.getElementById("iso_surface_checkbox").checked;
	sendFeedback("iso surface", status);
};

document.getElementById("zoom_in").onclick = function () {
	zoom(+0.1);
}

document.getElementById("zoom_out").onclick = function () {
	zoom(-0.1);
}

document.getElementById("plus_x").onclick = function () {
	rotate(+5, 0, 0);
}

document.getElementById("minus_x").onclick = function () {
	rotate(-5, 0, 0);
}

document.getElementById("plus_y").onclick = function () {
	rotate(0, +5, 0);
}

document.getElementById("minus_y").onclick = function () {
	rotate(0, -5, 0);
}

document.getElementById("plus_z").onclick = function () {
	rotate(0, 0, +5);
}

document.getElementById("minus_z").onclick = function () {
	rotate(0, 0, -5);
}

document.getElementById("ctl_left").onclick = function () {
	move([-0.05, 0, 0]);
}

document.getElementById("ctl_right").onclick = function () {
	move([+0.05, 0, 0]);
}

document.getElementById("ctl_up").onclick = function () {
	move([0, -0.05, 0]);
}

document.getElementById("ctl_down").onclick = function () {
	move([0, +0.05, 0]);
}


document.getElementById("zoom_in_5").onclick = function () {
	zoom(+0.5);
}

document.getElementById("zoom_out_5").onclick = function () {
	zoom(-0.5);
}

document.getElementById("plus_x_5").onclick = function () {
	rotate(+25, 0, 0);
}

document.getElementById("minus_x_5").onclick = function () {
	rotate(-25, 0, 0);
}

document.getElementById("plus_y_5").onclick = function () {
	rotate(0, +25, 0);
}

document.getElementById("minus_y_5").onclick = function () {
	rotate(0, -25, 0);
}

document.getElementById("plus_z_5").onclick = function () {
	rotate(0, 0, +25);
}

document.getElementById("minus_z_5").onclick = function () {
	rotate(0, 0, -25);
}

document.getElementById("ctl_left_5").onclick = function () {
	move([-0.25, 0, 0]);
}

document.getElementById("ctl_right_5").onclick = function () {
	move([+0.25, 0, 0]);
}

document.getElementById("ctl_up_5").onclick = function () {
	move([0, -0.25, 0]);
}

document.getElementById("ctl_down_5").onclick = function () {
	move([0, +0.25, 0]);
}

function controls_checkbox_click() {
	if (document.getElementById("controls_checkbox").checked)
		document.getElementById("controls").style.display = 'initial';
	else
		document.getElementById("controls").style.display = 'none';
}

function updateWireframe() {
	let color = RGB2HTML(
		255 - parseInt(background_color.jscolor.rgb[0]),
		255 - parseInt(background_color.jscolor.rgb[1]),
		255 - parseInt(background_color.jscolor.rgb[2]));
	let enabled = document.getElementById("wireframe_checkbox").checked;
	renderer.setWireframe(enabled, color);
}

/**
 * 
 * @param {IsaacResponse} response 
 */
function aoSetValues(response) {
	
	if (response.hasOwnProperty("ao isEnabled")) {

		document.getElementById("ao_checkbox").checked = response["ao isEnabled"];
		lastIsaacState["ao isEnabled"] = response["ao isEnabled"];
	}
	if (response.hasOwnProperty("ao weight")) {
		let weightEl = document.getElementById("ao_weight");

		document.getElementById("ao_weight").innerHTML = weightEl.value;
		lastIsaacState["ao_weight"] = response["ao weight"];;
	}
}

/**
 * called when wheight value is set or isEnabled is changed
 */
function aoUpdate() {
	let weight = parseFloat(document.getElementById("ao_weight").value);
	let status = document.getElementById("ao_checkbox").checked;

	let ao_object = {
		"isEnabled": status,
		"weight": weight
	};
	sendFeedback("ao", ao_object);
}