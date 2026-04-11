export interface DateRange {
    startDate: string;
    endDate: string;
}

export interface DateRangeDomain {
    min: Date;
    max: Date;
    minDate: string;
    maxDate: string;
    defaultRange: DateRange;
}

export type DateRangePresetUnit = 'days' | 'months' | 'years';

export interface DateRangePreset {
    label: string;
    amount: number;
    unit: DateRangePresetUnit;
}

export const toDateOnly = (value: Date): Date => {
    const next = new Date(value);
    next.setHours(0, 0, 0, 0);
    return next;
};

export const addDays = (value: Date, days: number): Date => {
    const next = new Date(value);
    next.setDate(next.getDate() + days);
    return toDateOnly(next);
};

export const addMonths = (value: Date, months: number): Date => {
    const current = toDateOnly(value);
    const target = new Date(current.getFullYear(), current.getMonth() + months, 1);
    const lastDayOfTargetMonth = new Date(target.getFullYear(), target.getMonth() + 1, 0).getDate();
    const next = new Date(
        target.getFullYear(),
        target.getMonth(),
        Math.min(current.getDate(), lastDayOfTargetMonth),
    );
    return toDateOnly(next);
};

export const addYears = (value: Date, years: number): Date => addMonths(value, years * 12);

export const formatIsoDate = (value: Date): string => {
    const year = value.getFullYear();
    const month = `${value.getMonth() + 1}`.padStart(2, '0');
    const day = `${value.getDate()}`.padStart(2, '0');
    return `${year}-${month}-${day}`;
};

export const parseIsoDate = (value: string): Date | null => {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
        return null;
    }

    const parsed = new Date(`${value}T00:00:00`);
    if (Number.isNaN(parsed.getTime())) {
        return null;
    }

    return toDateOnly(parsed);
};

export const clampDateToDomain = (value: Date, domain: DateRangeDomain): Date => {
    if (value < domain.min) {
        return domain.min;
    }
    if (value > domain.max) {
        return domain.max;
    }
    return value;
};

export const buildPresetDateRange = (
    anchorDate: Date,
    preset: DateRangePreset,
    domain: DateRangeDomain,
): DateRange => {
    const endDate = clampDateToDomain(toDateOnly(anchorDate), domain);

    let startCandidate: Date;
    switch (preset.unit) {
        case 'days':
            startCandidate = addDays(endDate, -(preset.amount - 1));
            break;
        case 'months':
            startCandidate = addDays(addMonths(endDate, -preset.amount), 1);
            break;
        case 'years':
            startCandidate = addDays(addYears(endDate, -preset.amount), 1);
            break;
        default:
            startCandidate = endDate;
            break;
    }

    const startDate = clampDateToDomain(startCandidate, domain);

    return {
        startDate: formatIsoDate(startDate > endDate ? endDate : startDate),
        endDate: formatIsoDate(endDate),
    };
};

export const buildIndicesDateRangeDomain = (
    historyWindowYears: number = 10,
    defaultRangeYears: number = 1,
): DateRangeDomain => {
    const max = toDateOnly(new Date());
    const min = toDateOnly(new Date(max.getFullYear() - historyWindowYears, max.getMonth(), max.getDate()));
    const defaultStartCandidate = addDays(
        new Date(max.getFullYear() - defaultRangeYears, max.getMonth(), max.getDate()),
        1,
    );
    const defaultStart = defaultStartCandidate < min ? min : defaultStartCandidate;

    return {
        min,
        max,
        minDate: formatIsoDate(min),
        maxDate: formatIsoDate(max),
        defaultRange: {
            startDate: formatIsoDate(defaultStart),
            endDate: formatIsoDate(max),
        },
    };
};
