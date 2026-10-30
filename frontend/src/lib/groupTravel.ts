import type { Friend } from '../types';

type TravelPerson = { id: string; name: string; originId: string | null };
type TravelOrigin = Friend;

// Keep one model surface per household, but count every confirmed-free person
// once in the objective. Names are labels; only stable IDs establish the link.
export function deriveGroupTravel(selection: { origins: TravelOrigin[]; freePeople: TravelPerson[] }) {
  const seen = new Set<string>();
  const available = selection.freePeople.filter(person => {
    if (seen.has(person.id)) return false;
    seen.add(person.id);
    return true;
  });
  const origins = new Map(selection.origins.filter(origin => Number.isFinite(origin.lat) && Number.isFinite(origin.lng)).map(origin => [origin.id, origin]));
  const friends: Friend[] = [];
  const includedIndexes: number[] = [];
  const originIndexes = new Map<string, number>();
  const missingLocationNames: string[] = [];
  for (const person of available) {
    const origin = person.originId ? origins.get(person.originId) : undefined;
    if (!origin) { missingLocationNames.push(person.name); continue; }
    let index = originIndexes.get(origin.id);
    if (index === undefined) {
      index = friends.length;
      originIndexes.set(origin.id, index);
      friends.push({ ...origin, name: person.name });
    } else {
      friends[index].name += ` & ${person.name}`;
    }
    includedIndexes.push(index);
  }
  return { friends, includedIndexes, availableCount: available.length, coveredCount: includedIndexes.length, missingLocationNames };
}
