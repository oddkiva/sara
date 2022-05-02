<script>
	import { onMount } from 'svelte';
	import { default as Module } from '../components/test_image_dewarp_renderer.js';

	let files;
	let videoUrl;
	let video;
	let copyVideo = false;

	// UI interactions.
	let _stopVideo = false;
	let _resizeCanvas = false;
	let _hideMousePointer = false;

	// Our reference to the Emscripten module.
	let _emmodule;
	// Our reference to the GL canvas.
	let _emcanvas;
	let _emglctx;

	const setupVideo = (url) => {
		var playing = false;
		var timeupdate = false;

		video.autoplay = true;
		video.muted = true;
		video.loop = true;

		// Waiting for these 2 events ensures
		// there is data in the video

		video.addEventListener(
			'playing',
			function () {
				playing = true;
				checkReady();
			},
			true
		);

		video.addEventListener(
			'timeupdate',
			function () {
				timeupdate = true;
				checkReady();
			},
			true
		);

		video.src = url;
		video.play();

		function checkReady() {
			if (playing && timeupdate) {
				copyVideo = true;
			}
		}

		return video;
	};

	const updateTexture = (gl, texture, video) => {
		const level = 0;
		const internalFormat = gl.RGB;
		const srcFormat = gl.RGB;
		const srcType = gl.UNSIGNED_BYTE;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texImage2D(
			gl.TEXTURE_2D,
			level,
			internalFormat,
			1920,
			1080,
			level,
			srcFormat,
			srcType,
			video
		);
		// We have got no other choice but to keep regenerating mipmaps. Otherwise
		// we will see white artefacts when zooming out on videos.
		gl.generateMipmap(gl.TEXTURE_2D);
		console.log('update texture');
	};

	onMount(async () => {
		// Create dummy JSEvents.
		window.JSEvents = null;
		window.specialHTMLTargets = null;

		_emmodule = await Module({
			noInitialRun: true,
			canvas: _emcanvas
		});

		// Run WebGL.
		_emmodule.callMain();
		_emglctx = _emmodule.GL.currentContext.GLctx;

		// window.emmodule = _emmodule;
		// window.GLctx = _emglctx;

		setInterval(() => {
			if (files) {
				const texture = _emmodule.GL.textures.slice(-1)[0];
				updateTexture(_emglctx, texture, video);
			}
		}, 16);
	});

	$: {
		if (files) {
			const URL = window.URL || window.webkitURL;
			const newVideoUrl = URL.createObjectURL(files[0]);

			if (videoUrl !== newVideoUrl) {
				videoUrl = newVideoUrl;
				setupVideo(videoUrl);
			}

			_emcanvas.hidden = false;
		}

		if (typeof video !== 'undefined' && video !== null && videoUrl !== null) {
			if (_stopVideo) video.pause();
			else video.play();
		}
	}
</script>

<div class="w-full items-center">
	<video bind:this={video} hidden="true" />

	<input
		class="m-1 p-1 bg-gray-100 hover:bg-gray-500 rounded-md"
		accept="video/*"
		bind:files
		type="file"
	/>

	<input class="btn btn-blue" type="checkbox" bind:checked={_stopVideo} />Stop video
	<input class="btn btn-blue" type="checkbox" bind:checked={_resizeCanvas} />Resize canvas
	<input class="btn btn-blue" type="checkbox" bind:checked={_hideMousePointer} />Lock/hide mouse
	pointer &nbsp;&nbsp;&nbsp;
	<button
		class="m-1 p-1 bg-gray-100 hover:bg-gray-500 rounded-md"
		on:click={_emmodule.requestFullscreen(_hideMousePointer, _resizeCanvas)}>Fullscreen</button
	>

	<hr />

	<canvas class="p-8 items-center" bind:this={_emcanvas} width="640" height="480" />
</div>
