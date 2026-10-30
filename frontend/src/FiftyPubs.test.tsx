import { useState } from 'react';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FiftyPubsDrawer, FiftyPubsRecommendations } from './FiftyPubs';
import type { PubDataset, RankedPub } from './lib/fiftyPubs';

const rankedPub = (id: string, rank: number, name: string, score: number | null): RankedPub => ({
  id, rank, name, score, neighbourhood: id === 'a' ? 'North quarter' : 'South quarter',
  address: `${rank} Test Road, London`, postcode: 'EX1 1AA', lat: 51.51, lng: -0.12,
  summary: `${name} has an original concise summary.`, sourceUrl: `https://editor.example/${id}`,
  addressSourceUrl: `https://venue.example/${id}`, coordinateSourceUrl: 'https://maps.example/place',
  images: [1, 2, 3, 4].map(i => ({ url: `https://images.example/${id}-${i}.jpg`, sourceUrl: `https://images.example/source-${i}`, credit: `Photographer ${i}`, license: 'CC BY-SA 4.0' })),
  tags: ['Independent'], journeys: score === null ? [null, null] : [score, score + 10],
  average: score, longest: score === null ? null : score + 10, sampledDistanceMetres: 50
});
const pubs = [rankedPub('a', 20, 'Amber Arms', 15), rankedPub('b', 2, 'Birch Bell', 20), rankedPub('c', 1, 'Cedar Crown', null)];
const dataset: PubDataset = { article: { title: 'The 50 best pubs in London', url: 'https://editor.example/list', checkedAt: '2026-10-30' }, pubs };
const friends = [{ id: 'one', name: 'Pair One', lat: 51.55, lng: -0.1 }, { id: 'two', name: 'Pair Two', lat: 51.52, lng: -0.2 }];
function drawerProps() {
  return { open: true, pubs, dataset, activeId: null, friends, included: [true, false], loading: false, onSelect: vi.fn(), onClose: vi.fn() };
}
afterEach(cleanup);

describe('50 pubs drawer', () => {
  it('does not render until opened and focuses its close button when shown', () => {
    const props = drawerProps();
    const view = render(<FiftyPubsDrawer {...props} open={false} />);
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
    view.rerender(<FiftyPubsDrawer {...props} />);
    expect(screen.getByRole('button', { name: 'Close pub details' })).toHaveFocus();
  });

  it('searches names, neighbourhoods and addresses case-insensitively and reports empty results', () => {
    render(<FiftyPubsDrawer {...drawerProps()} />);
    const search = screen.getByRole('textbox', { name: 'Search the 50 pubs' });
    fireEvent.change(search, { target: { value: 'AMBER' } });
    expect(screen.getByRole('button', { name: /Amber Arms/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Birch Bell/ })).not.toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'south quarter' } });
    expect(screen.getByRole('button', { name: /Birch Bell/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Amber Arms/ })).not.toBeInTheDocument();
    fireEvent.change(search, { target: { value: '20 Test Road' } });
    expect(screen.getByRole('button', { name: /Amber Arms/ })).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'not a pub' } });
    expect(screen.getByText('No pubs match that search.')).toBeInTheDocument();
  });

  it('switches between journey fit and editorial order while preserving printed ranks', () => {
    const { container } = render(<FiftyPubsDrawer {...drawerProps()} />);
    const names = () => Array.from(container.querySelectorAll('.fifty-list button strong')).map(node => node.textContent);
    expect(names()).toEqual(['Amber Arms', 'Birch Bell', 'Cedar Crown']);
    fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value: 'editorial' } });
    expect(names()).toEqual(['Cedar Crown', 'Birch Bell', 'Amber Arms']);
    expect(Array.from(container.querySelectorAll('.fifty-list-rank')).map(node => node.textContent)).toEqual(['1', '2', '20']);
    fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value: 'fit' } });
    expect(names()).toEqual(['Amber Arms', 'Birch Bell', 'Cedar Crown']);
  });

  it('opens venue detail with three attributed images, exact location and editorial review link, then returns to list', () => {
    function Flow() {
      const [activeId, onSelect] = useState<string | null>(null);
      return <FiftyPubsDrawer {...drawerProps()} activeId={activeId} onSelect={onSelect} />;
    }
    render(<Flow />);
    fireEvent.click(screen.getByRole('button', { name: /Amber Arms/ }));
    expect(screen.getByRole('complementary', { name: 'Amber Arms details' })).toBeInTheDocument();
    expect(screen.getByText('#20 in Time Out’s list')).toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(3);
    expect(screen.getByRole('link', { name: 'Photographer 1' })).toHaveAttribute('href', 'https://images.example/source-1');
    expect(screen.getByRole('link', { name: /Read the full Time Out review/ })).toHaveAttribute('href', pubs[0].sourceUrl);
    expect(screen.getByRole('link', { name: /Open location in Maps/ })).toHaveAttribute('href', 'https://www.google.com/maps/search/?api=1&query=51.51,-0.12');
    expect(screen.getByText('Pair Two · excluded')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'All 50 pubs' }));
    expect(screen.getByRole('textbox', { name: 'Search the 50 pubs' })).toBeInTheDocument();
  });

  it('closes by explicit button or Escape and restores previous focus', () => {
    const props = drawerProps();
    const button = document.createElement('button'); document.body.append(button); button.focus();
    const view = render(<FiftyPubsDrawer {...props} />);
    fireEvent.click(screen.getByRole('button', { name: 'Close pub details' }));
    expect(props.onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(props.onClose).toHaveBeenCalledTimes(2);
    view.rerender(<FiftyPubsDrawer {...props} open={false} />);
    expect(button).toHaveFocus();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(props.onClose).toHaveBeenCalledTimes(2);
    button.remove();
  });

  it('replaces a broken image with a labelled fallback but keeps its attribution', () => {
    render(<FiftyPubsDrawer {...drawerProps()} activeId='a' />);
    fireEvent.error(screen.getAllByRole('img')[0]);
    expect(screen.getByText('Photo unavailable')).toBeInTheDocument();
    expect(screen.getAllByRole('img')).toHaveLength(2);
    expect(screen.getByRole('link', { name: 'Photographer 1' })).toBeInTheDocument();
  });

  it('supports a single image and clearly describes absent photos and model coverage', () => {
    const one = { ...pubs[0], images: pubs[0].images.slice(0, 1) };
    const props = drawerProps();
    const view = render(<FiftyPubsDrawer {...props} pubs={[one]} activeId='a' />);
    expect(screen.getAllByRole('img')).toHaveLength(1);
    view.rerender(<FiftyPubsDrawer {...props} pubs={[{ ...pubs[2], images: [] }]} activeId='c' />);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText('No verified reusable photo is available.')).toBeInTheDocument();
    expect(screen.getByText(/Outside the current travel-time model coverage/)).toBeInTheDocument();
  });
});

describe('50 pub recommendations', () => {
  it('shows only the best three covered pubs and selects a stable venue ID', () => {
    const onSelect = vi.fn(), onBrowse = vi.fn();
    const choices = [...pubs.slice(0, 2), rankedPub('d', 49, 'Delta Duck', 25), rankedPub('e', 50, 'Elm Eagle', 30), pubs[2]];
    render(<FiftyPubsRecommendations pubs={choices} loading={false} error={null} onSelect={onSelect} onBrowse={onBrowse} />);
    const region = screen.getByRole('region', { name: 'Recommended pubs' });
    expect(within(region).getByRole('button', { name: /Amber Arms/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Elm Eagle|Cedar Crown/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Delta Duck/ }));
    expect(onSelect).toHaveBeenCalledWith('d');
    fireEvent.click(screen.getByRole('button', { name: /Explore all/ }));
    expect(onBrowse).toHaveBeenCalledOnce();
  });
});
