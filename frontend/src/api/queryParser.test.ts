import * as fs from 'fs';
import * as path from 'path';
import { parseQueryFile } from './queryParser';


function assertDeepEqual(actual: any, expected: any, path = '') {
  if (actual === expected) return;

  if (typeof actual !== typeof expected) {
    throw new Error(`Type mismatch at ${path}: actual ${typeof actual}, expected ${typeof expected}`);
  }

  if (actual && expected && typeof actual === 'object') {
    if (Array.isArray(actual) !== Array.isArray(expected)) {
      throw new Error(`Array mismatch at ${path}`);
    }

    const actualKeys = Object.keys(actual);
    const expectedKeys = Object.keys(expected);

    if (actualKeys.length !== expectedKeys.length) {
      throw new Error(`Keys length mismatch at ${path}: actual keys [${actualKeys}], expected keys [${expectedKeys}]`);
    }

    for (const key of actualKeys) {
      assertDeepEqual(actual[key], expected[key], path ? `${path}.${key}` : key);
    }
    return;
  }

  throw new Error(`Value mismatch at ${path}: actual "${actual}", expected "${expected}"`);
}

function runTests() {
  console.log('Starting Query File Parser Tests...\n');

  const testCases = [
    {
      filename: 'query-p1-1-kis.txt',
      expected: {
        query_id: 'p1-1',
        task_type: 'kis',
        request: {
          query_id: 'p1-1',
          query: 'Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh, tiếp theo đến cảnh quay tòa nhà Bitexco. Một vài cảnh sau đó chuyển qua quay hình ảnh hồ gươm tại Hà Nội.',
          top_k: 5,
        },
      },
    },
    {
      filename: 'query-p1-2-kis.txt',
      expected: {
        query_id: 'p1-2',
        task_type: 'kis',
        request: {
          query_id: 'p1-2',
          query: 'Một người đàn ông đang trả lời phỏng vấn trong một lễ hội. Phía sau người đàn ông này là một vật trang trí có hình dáng con chim màu tím.',
          top_k: 5,
        },
      },
    },
    {
      filename: 'query-p1-3-qa.txt',
      expected: {
        query_id: 'p1-3',
        task_type: 'qa',
        request: {
          query_id: 'p1-3',
          scene: 'Đây là phần giới thiệu phần thưởng cho một cuộc thi được thiết kế với nền xanh, chữ trắng phủ lên video giới thiệu cuộc thi. Tổng có 18 giải thưởng chính cho cuộc thi.',
          question: 'tổng giá trị của các giải thưởng chính là bao nhiêu?',
          top_k: 3,
        },
      },
    },
    {
      filename: 'query-p1-4-qa.txt',
      expected: {
        query_id: 'p1-4',
        task_type: 'qa',
        request: {
          query_id: 'p1-4',
          scene: 'Một bác sĩ tóc bạc, đeo kính đang trả lời phỏng vấn. Sau đó là cảnh vị bác sĩ này đang nói chuyện với một bệnh nhân nữ quay lưng lại với camera mặc áo đen. Trên bàn có 2 cây bút. Bên trái áo của bác sĩ có thêu dòng chữ màu đỏ gồm tên, học hàm, học vị của vị bác sĩ này.',
          question: 'họ và tên đầy đủ của bác sĩ là gì?',
          top_k: 3,
        },
      },
    },
    {
      filename: 'query-p1-5-trake.txt',
      expected: {
        query_id: 'p1-5',
        task_type: 'trake',
        request: {
          query_id: 'p1-5',
          events: [
            'Người đầu bếp cho cá vào một tô màu trắng. Hãy lấy khoảnh khắc con cá cuoói cùng rớt khỏi dĩa',
            'Người đầu bếp đổ bột vào một tô cá để chiên. Hãy lấy khoảnh khắc đầu tiên mà chiếc đũa của người đầu bếp chạm vào cá để trộn bột.',
            'Tiếp theo, người đầu bếp này dùng đũa để kiểm tra độ nóng của dầu. Hãy lấy khoảnh khắc đầu tiên chiếc đũa được nhấc ra khỏi dầu.',
          ],
          top_k: 3,
        },
      },
    },
    {
      filename: 'query-p1-6-trake.txt',
      expected: {
        query_id: 'p1-6',
        task_type: 'trake',
        request: {
          query_id: 'p1-6',
          events: [
            'Một người  đang cắt đôi ổ bánh mì có rắc mè rồi đem nướng trên chảo. Hãy lấy khoảnh khắc chiếc dao cắt qua hoàn toàn chiếc bánh.',
            'Sau đó người này rắc bột lên những miếng thịt, trong quá trình này người đầu bếp lật những miếng thịt để rắc bột đều hai mặt. Hãy lấy khoảnh khắc đầu tiên người đầu bếp này buông tay khỏi miếng thịt sau khi lật miếng thịt đầu tiên.',
            'Các miếng thịt sau đó được đem đi áp chảo cùng với bơ (3 ngang 1 dọc theo chiều của camera). Hãy lấy khoảnh khắc đầu tiên người đầu bếp cầm vào chảo để nhấc lên đảo bơ đều xung quanh.',
          ],
          top_k: 3,
        },
      },
    },
  ];

  // Base directory for references
  const refsDir = path.resolve(process.cwd(), '../docs/References/queries-pack-0');

  let passed = 0;
  let failed = 0;

  for (const tc of testCases) {
    const filePath = path.join(refsDir, tc.filename);
    try {
      console.log(`Testing file: ${tc.filename}`);
      const content = fs.readFileSync(filePath, 'utf-8');
      const parsed = parseQueryFile(tc.filename, content);

      assertDeepEqual(parsed, tc.expected);
      console.log(`  [PASS] parsed query matches expected structure\n`);
      passed++;
    } catch (err: any) {
      console.error(`  [FAIL] ${err.message}\n`);
      failed++;
    }
  }

  // Defensive validation error tests
  console.log('Testing defensive error handling...');
  
  try {
    parseQueryFile('', 'some content');
    console.error('  [FAIL] Expected empty filename to throw error');
    failed++;
  } catch (err: any) {
    console.log(`  [PASS] Empty filename throw error: "${err.message}"`);
    passed++;
  }

  try {
    parseQueryFile('query-p1-1-invalidtype.txt', 'some content');
    console.error('  [FAIL] Expected invalid type to throw error');
    failed++;
  } catch (err: any) {
    console.log(`  [PASS] Invalid type throw error: "${err.message}"`);
    passed++;
  }

  try {
    parseQueryFile('query-p1-3-qa.txt', 'No separator here');
    console.error('  [FAIL] Expected missing "Hỏi" separator in QA to throw error');
    failed++;
  } catch (err: any) {
    console.log(`  [PASS] Missing "Hỏi" separator in QA throw error: "${err.message}"`);
    passed++;
  }

  try {
    parseQueryFile('query-p1-5-trake.txt', 'No event prefixes here');
    console.error('  [FAIL] Expected empty events in Trake to throw error');
    failed++;
  } catch (err: any) {
    console.log(`  [PASS] Empty events in Trake throw error: "${err.message}"`);
    passed++;
  }

  console.log(`\nTests Summary: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exit(1);
  } else {
    console.log('All tests passed successfully!');
  }
}

runTests();
