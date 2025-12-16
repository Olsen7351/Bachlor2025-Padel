const StatBar = ({ label, value, max }: { label: string; value: number; max: number }) => (
    <div className="space-y-1">
        <div className="flex justify-between text-sm text-white/80">
            <span>{label}</span>
            <span>{value}</span>
        </div>
        <div className="h-2 bg-white/10 rounded">
            <div
                className="h-2 rounded bg-gradient-to-r from-cyan-400 to-blue-500 transition-all"
                style={{ width: `${(value / max) * 100}%` }}
            />
        </div>
    </div>
);

export default StatBar;