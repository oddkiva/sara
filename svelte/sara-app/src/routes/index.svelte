<script>
	import { pokemons } from '../stores/photostore.js';
	import PokemonCard from '../components/pokemonCard.svelte';

	const greetings = 'SvelteKit Sara';

	let searchTerm = '';
	let filteredPokemons;

	$: {
		if (!searchTerm) {
			filteredPokemons = $pokemons;
		} else {
			filteredPokemons = $pokemons.filter((pokemon) =>
				pokemon.name.toLowerCase().includes(searchTerm.toLowerCase())
			);
		}
	}
</script>

<svelte:head>
	<title>SvelteKit Sara</title>
</svelte:head>

<h1 class="text-4xl text-center uppercase my-8">{greetings}</h1>

<input
	class="w-full mx-4 text-center"
	bind:value={searchTerm}
	type="text"
	placeholder="Search Pokemon..."
/>

<div class="mx-4 py-4 grid gap-4 md:grid-cols-2 grid-cols-1">
	{#each filteredPokemons as pokemon}
		<PokemonCard {pokemon} />
	{/each}
</div>
