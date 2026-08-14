# Frontend Architecture & State Management (FE1 & FE2 Alignment)

Tài liệu này chốt phương án quản lý trạng thái chung (Shared State Management) dành cho phái đoàn phát triển **FE1** và **FE2** thuộc dự án **SIU Collective Retrieval System** (Issue #97).

---

## 1. Phương án State Management được chọn

- **Approach**: **React Context + `useReducer`**
- **Vị trí file**: [`frontend/src/context/SearchContext.tsx`](file:///d:/SIU_Collective/frontend/src/context/SearchContext.tsx)
- **Lý do chọn**:
  1. Không phát sinh thêm thư viện bên ngoài (Zero external dependencies như Zustand, Redux).
  2. Sẵn có trong React 19, tương thích hoàn hảo với Vite và TypeScript.
  3. Đủ nhẹ cho scope hiện tại, tránh over-engineering nhưng vẫn đủ khả năng mở rộng khi làm các task #98 (Pagination) và #99 (Real endpoints).

---

## 2. Bảng mô tả các State chung (Shared State)

Các thuộc tính state chung được quản lý trong `SearchState`:

| State Field | Type | Description |
| :--- | :--- | :--- |
| `taskType` | `'KIS' \| 'Trake' \| 'Q&A'` | Loại tác vụ retrieval đang chọn |
| `searchMode` | `'manual' \| 'competition'` | Chế độ tìm kiếm thủ công hoặc đọc file hàng loạt |
| `currentQuery` | `KisRequest \| TrakeRequest \| QaRequest \| null` | Thông tin query hiện tại |
| `currentResults` | `KisResponse \| QaResponse \| unknown \| null` | Kết quả truy vấn trả về từ API/Mock |
| `isLoading` | `boolean` | Trạng thái loading khi thực hiện truy vấn |
| `error` | `string \| null` | Thông báo lỗi khi xảy ra sự cố |
| `selectedItem` | `{ videoId, frameIdx?, timestampSeconds? } \| null` | Item/Video được chọn để preview trên Player |
| `batchQueueStatus` | `'idle' \| 'running' \| 'paused' \| 'completed'` | Trạng thái hàng chờ xử lý trong Competition Mode |

---

## 3. Hướng dẫn sử dụng dành cho FE1 và FE2

### A. Bọc Provider tại ứng dụng (`App.tsx`)
`SearchProvider` đã được bọc bên ngoài `BrowserRouter` trong `App.tsx`:
```tsx
import { SearchProvider } from '@/context/SearchContext';

function App() {
  return (
    <SearchProvider>
      <BrowserRouter>
        ...
      </BrowserRouter>
    </SearchProvider>
  );
}
```

### B. Sử dụng custom hook `useSearchContext` trong Components
FE1 và FE2 khi viết components chỉ cần gọi hook `useSearchContext`:

```tsx
import React from 'react';
import { useSearchContext } from '@/context/SearchContext';

export const ExampleComponent: React.FC = () => {
  const { state, setTaskType, setCurrentQuery, setIsLoading } = useSearchContext();

  const handleSelectTask = (type: 'KIS' | 'Trake' | 'Q&A') => {
    setTaskType(type);
  };

  return (
    <div>
      <p>Current Task: {state.taskType}</p>
      <p>Loading: {state.isLoading ? 'Yes' : 'No'}</p>
    </div>
  );
};
```

---

## 4. Quy tắc tránh Conflict khi Merge (FE1 & FE2 Alignment)

1. **Khi thêm State Field mới**: Thêm vào `SearchState` trong `SearchContext.tsx` và khai báo action tương ứng trong `SearchAction`.
2. **Không mutate State trực tiếp**: Luôn sử dụng helper methods hoặc `dispatch`.
3. **Phân chia công việc tiếp theo**:
   - FE1: Tiếp tục với Task #95 / #96 (Real endpoints, bilingual query handling).
   - FE2: Tiếp tục với Task #98 / #99 (Result pagination & state synchronization).
