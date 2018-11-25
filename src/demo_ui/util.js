export function isError(value) {
  return value instanceof Error && typeof value.message !== 'undefined';
}


export function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}
