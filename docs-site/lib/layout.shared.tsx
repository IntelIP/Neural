import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="neural-wordmark">
          <span className="neural-mark" aria-hidden="true">
            N
          </span>
          Neural SDK
        </span>
      ),
    },
    links: [
      {
        text: 'GitHub',
        url: 'https://github.com/IntelIP/Neural',
        external: true,
      },
      {
        text: 'Support',
        url: 'mailto:hudson@intelip.co',
        external: true,
      },
    ],
  };
}
