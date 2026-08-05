"use client";

interface ECGLeadSelectorProps {
  leads: string[];
  selectedLead: string;
  onSelectLead: (lead: string) => void;
}

export function ECGLeadSelector({
  leads,
  selectedLead,
  onSelectLead,
}: ECGLeadSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-2">
      {leads.map((lead) => {
        const active = lead === selectedLead;

        return (
          <button
            key={lead}
            type="button"
            onClick={() => onSelectLead(lead)}
            className={`rounded-xl px-3 py-2 text-xs font-semibold transition ${
              active
                ? "bg-slate-950 text-white shadow-sm"
                : "bg-white text-slate-600 hover:bg-slate-100"
            }`}
          >
            {lead}
          </button>
        );
      })}
    </div>
  );
}
