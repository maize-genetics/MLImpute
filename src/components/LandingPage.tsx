import React from 'react';
import Icon from '@mdi/react';
import {
  mdiMapMarkerPath,
  mdiChartTimeline,
  mdiArrowRight,
  mdiGithub,
  mdiBookOpenVariant,
} from '@mdi/js';
import { isTauri } from '../platform';
import './LandingPage.css';

type PageType = 'landing' | 'imputation' | 'ps4g' | 'bed';

interface LandingPageProps {
  onNavigate: (page: PageType) => void;
}

interface FeatureCard {
  page: PageType;
  icon: string;
  title: string;
  description: string;
  tauriOnly?: boolean;
}

const GITHUB_REPO_URL = 'https://github.com/maize-genetics/grits';
// Documentation is published alongside this web app under /docs/ on the
// combined GitHub Pages site. The desktop (Tauri) build has no such path, so
// the link is only rendered for the web build.
const DOCS_URL = './docs/';

const features: FeatureCard[] = [
  {
    page: 'imputation',
    icon: mdiMapMarkerPath,
    title: 'Imputation',
    description:
      'Run machine learning models (KNN, BiMamba, ModernBERT) on PS4G haplotype data with optional HMM post-processing and configurable weighting schemes.',
    tauriOnly: true,
  },
  {
    page: 'ps4g',
    icon: mdiChartTimeline,
    title: 'PS4G Explorer',
    description:
      'Load and inspect PS4G haplotype files. View gamete summaries, browse raw data, and visualize chromosome matrices with interactive heatmaps.',
  },
  {
    page: 'bed',
    icon: mdiChartTimeline,
    title: 'BED Explorer',
    description:
      'Analyze BED imputation output files. Explore parent assignments, view summary statistics, and visualize results with chromosome-level heatmaps.',
  },
];

const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  const visibleFeatures = features.filter(
    (f) => !f.tauriOnly || isTauri
  );

  return (
    <div className="landing-page-scroll">
      <div className="landing-page-inner">
        <section className="landing-hero">
          <h1 className="landing-hero-title">GRITS</h1>
          <p className="landing-hero-tagline">
            <strong>G</strong>enetic <strong>R</strong>ecombination{' '}
            <strong>I</strong>mputation <strong>T</strong>ool <strong>S</strong>et
          </p>
          <p className="landing-hero-description">
            GRITS combines multiple ML approaches, including state space models,
            transformers, and classical methods, to impute missing haplotype data
            from PS4G files and produce extended BED output for downstream analysis.
          </p>
          <div className="landing-hero-links">
            <a
              className="landing-github-link"
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={1} />
              View on GitHub
            </a>
            {!isTauri && (
              <a className="landing-github-link" href={DOCS_URL}>
                <Icon path={mdiBookOpenVariant} size={1} />
                Documentation
              </a>
            )}
          </div>
        </section>

        <section className="landing-features">
          <h2 className="landing-features-heading">Get Started</h2>
          <div className="landing-features-grid">
            {visibleFeatures.map((feature) => (
              <button
                key={feature.page}
                className="landing-feature-card"
                onClick={() => onNavigate(feature.page)}
              >
                <div className="landing-feature-header">
                  <div className="landing-feature-icon">
                    <Icon path={feature.icon} size={0.95} />
                  </div>
                  <h3 className="landing-feature-title">{feature.title}</h3>
                </div>
                <p className="landing-feature-description">
                  {feature.description}
                </p>
                <span className="landing-feature-action">
                  Open <Icon path={mdiArrowRight} size={0.7} />
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default LandingPage;
