<script>
	import { pokemons } from '../stores/photostore.js';

	export let pokemon;
</script>

<div
	class="p-6 bg-gray-100 text-gray-800 rounded-md shadow-sm
      hover:shadow-md flex flex-col items-center"
>
	<img class="h-40 w-40" src={pokemon.image} alt={pokemon.name} />
	<h2 class="uppercase">{pokemon.name}</h2>
</div>
