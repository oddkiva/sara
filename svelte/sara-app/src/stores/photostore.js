import { writable } from 'svelte/store';

export const pokemons = writable([]);

const maxNumPokemons = 150;

const fetchPokemons = async (num) => {
	const url = `https://pokeapi.co/api/v2/pokemon?limit=${num}`;
	const res = await fetch(url);
	const data = await res.json();
	console.log(data);
	const loadedPokemons = data.results.map((data, index) => {
		return {
			name: data.name,
			id: index + 1,
			image: `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${
				index + 1
			}.png`
		};
	});

	pokemons.set(loadedPokemons);
};

fetchPokemons(maxNumPokemons);
