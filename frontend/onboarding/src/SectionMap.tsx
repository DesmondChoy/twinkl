import { useEffect, useState } from "react";

export interface SectionMapItem {
  id: string;
  label: string;
  description: string;
}

interface SectionMapProps {
  description: string;
  eyebrow: string;
  navigationLabel: string;
  sections: readonly SectionMapItem[];
  title: string;
}

export default function SectionMap({
  description,
  eyebrow,
  navigationLabel,
  sections,
  title,
}: SectionMapProps) {
  const [activeSection, setActiveSection] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    setActiveSection(sections[0]?.id ?? "");

    const updateActiveSection = () => {
      const marker = Math.min(160, window.innerHeight * 0.24);
      let nextSection = sections[0]?.id ?? "";
      let hasLayout = false;

      sections.forEach((section) => {
        const target = document.getElementById(section.id);
        if (!target) return;
        const bounds = target.getBoundingClientRect();
        hasLayout ||= bounds.height > 0 || bounds.top !== 0;
        if (bounds.top <= marker) nextSection = section.id;
      });

      if (hasLayout) setActiveSection(nextSection);
    };

    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    window.addEventListener("resize", updateActiveSection);
    return () => {
      window.removeEventListener("scroll", updateActiveSection);
      window.removeEventListener("resize", updateActiveSection);
    };
  }, [sections]);

  return (
    <div className="section-map">
      <div className="section-map__intro">
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      <nav aria-label={navigationLabel}>
        <ol>
          {sections.map((section, index) => (
            <li key={section.id}>
              <a
                className="section-map__link"
                href={`#${section.id}`}
                aria-current={
                  activeSection === section.id ? "location" : undefined
                }
                aria-label={section.label}
                onClick={() => setActiveSection(section.id)}
              >
                <span aria-hidden="true">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span>
                  <strong>{section.label}</strong>
                  <small>{section.description}</small>
                </span>
              </a>
            </li>
          ))}
        </ol>
      </nav>
    </div>
  );
}
