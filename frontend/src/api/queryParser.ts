export interface KisRequest {
  query_id: string;
  query: string;
  top_k: number;
}

export interface TrakeRequest {
  query_id: string;
  events: string[];
  top_k: number;
}

export interface QaRequest {
  query_id: string;
  scene: string;
  question: string;
  top_k: number;
}

export interface ParsedQuery {
  query_id: string;
  task_type: 'kis' | 'trake' | 'qa';
  request: KisRequest | TrakeRequest | QaRequest;
}

/**
 * Parses query file content based on filename pattern and task type.
 * 
 * Filename pattern: query-{phase}-{id}-{task_type}.txt (e.g., query-p1-5-trake.txt)
 * - task_type: last element before .txt -> 'kis' | 'trake' | 'qa'
 * - query_id: combination of 2nd and 3rd elements -> 'p1-5'
 */
export function parseQueryFile(filename: string, content: string): ParsedQuery {
  if (!filename) {
    throw new Error('Filename is required');
  }
  if (content === undefined || content === null) {
    throw new Error('Content is required');
  }

  // Extract base name from path to handle full paths correctly
  const baseName = filename.split(/[/\\]/).pop() || filename;
  const cleanName = baseName.replace(/\.txt$/, '');
  const parts = cleanName.split('-');

  if (parts.length < 4) {
    throw new Error(`Invalid query filename format: "${filename}". Expected format: query-{phase}-{id}-{task_type}.txt`);
  }

  const task_type_raw = parts[parts.length - 1].toLowerCase();
  if (task_type_raw !== 'kis' && task_type_raw !== 'trake' && task_type_raw !== 'qa') {
    throw new Error(`Unsupported task type: "${task_type_raw}". Supported types are kis, trake, qa.`);
  }

  const task_type = task_type_raw as 'kis' | 'trake' | 'qa';
  const query_id = `${parts[1]}-${parts[2]}`;

  const trimmedContent = content.trim();

  if (task_type === 'kis') {
    if (!trimmedContent) {
      throw new Error('KIS query content cannot be empty');
    }
    return {
      query_id,
      task_type,
      request: {
        query_id,
        query: trimmedContent,
        top_k: 5,
      },
    };
  }

  if (task_type === 'qa') {
    const hoiIndex = trimmedContent.lastIndexOf('Hỏi');
    if (hoiIndex === -1) {
      throw new Error('Invalid Q&A query content format: missing "Hỏi" separator');
    }

    const scene = trimmedContent.substring(0, hoiIndex).trim();
    let question = trimmedContent.substring(hoiIndex).trim();

    if (question.startsWith('Hỏi')) {
      question = question.substring(3).trim();
    }

    if (!scene) {
      throw new Error('Q&A query scene description cannot be empty');
    }
    if (!question) {
      throw new Error('Q&A query question cannot be empty');
    }

    return {
      query_id,
      task_type,
      request: {
        query_id,
        scene,
        question,
        top_k: 3,
      },
    };
  }

  // Trake
  const lines = trimmedContent.split(/\r?\n/);
  const events: string[] = [];

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;

    const match = trimmedLine.match(/^E\d+:\s*(.*)$/i);
    if (match) {
      const eventDesc = match[1].trim();
      if (eventDesc) {
        events.push(eventDesc);
      }
    }
  }

  if (events.length === 0) {
    throw new Error('Invalid Trake query content format: no valid event lines (e.g. "E1: ...") found');
  }

  return {
    query_id,
    task_type,
    request: {
      query_id,
      events,
      top_k: 3,
    },
  };
}
