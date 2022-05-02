<script>
	import { onMount } from 'svelte';
	import { default as Module } from '../components/test_image_dewarp_renderer.js';

	let files;
	let videoUrl;
	let video;
	let copyVideo = false;
	let renderVideo = false;

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
    // We have got no choice but to do this. Otherwise zooming out will fail for
    // videos.
    gl.generateMipmap(gl.TEXTURE_2D);
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

		window.emmodule = _emmodule;
		window.GLctx = _emglctx;

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
				console.log('setup video');
				videoUrl = newVideoUrl;
				renderVideo = false;
				setupVideo(videoUrl);
				setTimeout(() => (renderVideo = true), 0);
			}

			_emcanvas.hidden = false;
		}
	}
</script>

<div class="w-full m-8 items-center">
	<video bind:this={video} hidden="true" />
	<input class="btn btn-blue" accept="video/*" bind:files type="file" />
	<canvas class="p-8 items-center" bind:this={_emcanvas} width="640" height="480" />
</div>
