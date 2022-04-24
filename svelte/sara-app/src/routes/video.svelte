<script>
	import { onMount } from 'svelte';
	import { setupVideo, gltest } from '../components/utilities.js';
	import Emscripten from '../components/emscripten.svelte';

	let files;
	let videoUrl;
	let renderVideo = false;

	$: if (files) {
		const URL = window.URL || window.webkitURL;
		const newVideoUrl = URL.createObjectURL(files[0]);

		if (videoUrl !== newVideoUrl) {
			videoUrl = newVideoUrl;
			renderVideo = false;
			setTimeout(() => (renderVideo = true), 0);
		}

		let canvas = document.getElementById('gl-canvas');
		canvas.hidden = false;

		gltest(videoUrl);
	}
</script>

<div class="py-4 w-full text-center">
	<input class="btn btn-blue" accept="video/*" bind:files type="file" />
	<canvas class="p-8 items-center" hidden id="gl-canvas" width="640" height="480" display="none" />
</div>

<Emscripten />

<style>
	.btn {
		@apply font-bold py-2 px-4 rounded;
	}
	.btn-blue {
		@apply bg-blue-500 text-white;
	}
	.btn-blue:hover {
		@apply bg-blue-700;
	}
</style>
