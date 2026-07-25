import { VALUES, type ValueKey } from "./domain";

export function displayCoreValue(value: string): string {
  if (Object.hasOwn(VALUES, value)) {
    return VALUES[value as ValueKey].name;
  }
  return value
    .split("_")
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

export function displayWeekRange(start: string, end: string): string {
  const startDate = new Date(`${start}T00:00:00`);
  const endDate = new Date(`${end}T00:00:00`);
  const monthDay = new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
  });
  const day = new Intl.DateTimeFormat(undefined, { day: "numeric" });
  const year = new Intl.DateTimeFormat(undefined, { year: "numeric" });

  if (
    startDate.getFullYear() === endDate.getFullYear() &&
    startDate.getMonth() === endDate.getMonth()
  ) {
    return `${monthDay.format(startDate)}–${day.format(endDate)}, ${
      year.format(endDate)
    }`;
  }
  if (startDate.getFullYear() === endDate.getFullYear()) {
    return `${monthDay.format(startDate)}–${monthDay.format(endDate)}, ${
      year.format(endDate)
    }`;
  }
  return `${monthDay.format(startDate)}, ${year.format(startDate)}–${
    monthDay.format(endDate)
  }, ${year.format(endDate)}`;
}
