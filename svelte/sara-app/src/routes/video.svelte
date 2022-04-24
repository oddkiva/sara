<script>
	import { onMount } from 'svelte';
	import { setupVideo, gltest } from '../components/utilities.js';

	let files;
	let videoUrl;
	let renderVideo = false;
	let copyVideo = false;

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
	<!--
	{#if renderVideo}
		<video class="p-8" controls autoplay>
			<track kind="caption" />
			<source src={videoUrl} type="video/mp4" />
		</video>
	{/if}
  -->

	<canvas class="p-8 items-center" hidden id="gl-canvas" width="640" height="480" display="none" />
</div>

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
