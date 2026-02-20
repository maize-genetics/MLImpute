import React, { useState, useCallback } from 'react';
import Icon from '@mdi/react';
import { mdiMagnify } from '@mdi/js';
import { parsePositionInput } from '../utils/positionSearch';

interface PositionSearchProps {
  /** Called with the parsed numeric position when the user submits a search. */
  onSearch: (parsedPosition: number) => void;
  disabled?: boolean;
  placeholder?: string;
  inputId?: string;
}

const PositionSearch: React.FC<PositionSearchProps> = ({
  onSearch,
  disabled = false,
  placeholder = 'e.g. 3.1K; 3100; 3,100',
  inputId = 'position-search-input',
}) => {
  const [searchValue, setSearchValue] = useState('');

  const handleSubmit = useCallback(() => {
    const parsed = parsePositionInput(searchValue);
    if (parsed !== null) onSearch(parsed);
  }, [searchValue, onSearch]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') handleSubmit();
    },
    [handleSubmit],
  );

  return (
    <div className="position-search">
      <label htmlFor={inputId}>Go to position:</label>
      <div className="search-input-wrapper">
        <input
          id={inputId}
          type="text"
          placeholder={placeholder}
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="search-button"
          onClick={handleSubmit}
          title="Go to position"
          disabled={disabled}
        >
          <Icon path={mdiMagnify} size={0.7} />
        </button>
      </div>
    </div>
  );
};

export default PositionSearch;
