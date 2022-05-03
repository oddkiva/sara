<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { default as Module } from '../components/test_image_dewarp_renderer.js';

	let files: FileList | null;
	let videoUrl: string | null;
	let video: HTMLMediaElement | null;
	let videoInterval: ReturnType<typeof setInterval> | null;

	// UI interactions.
	let _pauseVideo = false;
	let _resizeCanvas = false;
	let _hideMousePointer = false;

	// Our reference to the Emscripten module.
	let _emmodule: Object | null;
	// Our reference to the GL canvas.
	let _emcanvas: HTMLCanvasElement | null;
	let _emglctx: WebGL2RenderingContext | null;

	const setupVideo = (url: string) => {
		if (video === null) {
			return null;
		}

		video.src = url;
		video.autoplay = true;
		video.loop = true;
		video.play().catch((e) => {
			console.log('Caught video exception: ' + e);
		});
	};

	const updateTexture = (
		gl: WebGL2RenderingContext,
		texture: WebGLTexture,
		video: HTMLCanvasElement
	) => {
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
	};

	onMount(async () => {
		// Create dummy JSEvents.
		window.JSEvents = null;
		window.specialHTMLTargets = null;

		// Run WebGL.
		_emmodule = await Module({
			noInitialRun: true,
			canvas: _emcanvas
		});
		_emmodule.callMain();
		_emglctx = _emmodule.GL.currentContext.GLctx;

		// window.emmodule = _emmodule;
		// window.GLctx = _emglctx;

		videoInterval = setInterval(() => {
			if (files) {
				const texture = _emmodule.GL.textures.slice(-1)[0];
				if (video !== null) {
					updateTexture(_emglctx, texture, video);
				}
			}
		}, 16);
	});

	onDestroy(async () => {
		if (videoInterval !== null) {
			clearInterval(videoInterval);
		}
		if (video !== null) {
			video.pause();
		}

		delete window.JSEvents;
		delete window.specialHTMLTargets;
		// delete window.emmodule;
		// delete window.GLctx;
	});

	$: {
		if (files) {
			const URL = window.URL || window.webkitURL;
			const newVideoUrl = URL.createObjectURL(files[0]);

			if (videoUrl !== newVideoUrl) {
				videoUrl = newVideoUrl;
				setupVideo(videoUrl);
			}

			if (_emcanvas !== null) {
				_emcanvas.hidden = false;
			}
		}

		if (typeof video !== 'undefined' && video !== null && videoUrl !== null) {
			if (_pauseVideo) {
				clearInterval(videoInterval);
				videoInterval = null;
				video.pause();
			} else {
				if (videoInterval === null) {
					videoInterval = setInterval(() => {
						if (files) {
							const texture = _emmodule.GL.textures.slice(-1)[0];
							updateTexture(_emglctx, texture, video);
						}
					}, 16);
				}
				video.play().catch((e) => {
					console.log('Caught video exception: ' + e);
				});
			}
		}
	}
</script>

<div class="w-full items-center">
	<video bind:this={video} hidden muted />

	<input
		class="m-1 p-1 bg-gray-100 hover:bg-gray-500 rounded-md"
		accept="video/*"
		bind:files
		type="file"
	/>

	<input type="checkbox" bind:checked={_pauseVideo} />Stop video
	<input type="checkbox" bind:checked={_resizeCanvas} />Resize canvas
	<input type="checkbox" bind:checked={_hideMousePointer} />Lock/hide mouse pointer
	&nbsp;&nbsp;&nbsp;
	<button
		class="m-1 p-1 bg-gray-100 hover:bg-gray-500 rounded-md"
		on:click={_emmodule.requestFullscreen(_hideMousePointer, _resizeCanvas)}>Fullscreen</button
	>

	<hr />

	<canvas class="p-8 items-center" bind:this={_emcanvas} width="640" height="480" />
</div>
