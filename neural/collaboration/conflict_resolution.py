"""
Conflict Resolution - Handles conflicts during concurrent editing.

Implements operational transformation and three-way merge for resolving edit conflicts.
"""

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from neural.exceptions import ConflictError


class OperationType(Enum):
    """Types of edit operations."""
    INSERT = 'insert'
    DELETE = 'delete'
    REPLACE = 'replace'


@dataclass
class EditOperation:
    """Represents a single edit operation."""
    type: OperationType
    position: int
    length: int
    content: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary."""
        return {
            'type': self.type.value,
            'position': self.position,
            'length': self.length,
            'content': self.content,
            'user_id': self.user_id,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditOperation':
        """Create operation from dictionary."""
        return cls(
            type=OperationType(data['type']),
            position=data['position'],
            length=data['length'],
            content=data.get('content'),
            user_id=data.get('user_id'),
            timestamp=data.get('timestamp')
        )


class ConflictResolver:
    """
    Resolves edit conflicts using operational transformation.
    
    Implements operational transformation (OT) for concurrent editing,
    allowing multiple users to edit the same document simultaneously.
    """
    
    def __init__(self):
        """Initialize conflict resolver."""
        self.operation_history: List[EditOperation] = []
    
    def transform_operations(
        self,
        op1: EditOperation,
        op2: EditOperation
    ) -> Tuple[EditOperation, EditOperation]:
        """
        Transform two concurrent operations using operational transformation.
        
        Parameters
        ----------
        op1 : EditOperation
            First operation
        op2 : EditOperation
            Second operation
            
        Returns
        -------
        Tuple[EditOperation, EditOperation]
            Transformed operations (op1', op2')
        """
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            return self._transform_insert_insert(op1, op2)
        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            return self._transform_insert_delete(op1, op2)
        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            op2_prime, op1_prime = self._transform_insert_delete(op2, op1)
            return op1_prime, op2_prime
        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            return self._transform_delete_delete(op1, op2)
        elif op1.type == OperationType.REPLACE or op2.type == OperationType.REPLACE:
            return self._transform_with_replace(op1, op2)
        
        return op1, op2
    
    def _transform_insert_insert(
        self,
        op1: EditOperation,
        op2: EditOperation
    ) -> Tuple[EditOperation, EditOperation]:
        """Transform two concurrent insert operations."""
        if op1.position < op2.position:
            op2_prime = EditOperation(
                type=op2.type,
                position=op2.position + len(op1.content or ''),
                length=op2.length,
                content=op2.content,
                user_id=op2.user_id,
                timestamp=op2.timestamp
            )
            return op1, op2_prime
        elif op1.position > op2.position:
            op1_prime = EditOperation(
                type=op1.type,
                position=op1.position + len(op2.content or ''),
                length=op1.length,
                content=op1.content,
                user_id=op1.user_id,
                timestamp=op1.timestamp
            )
            return op1_prime, op2
        else:
            if (op1.user_id or '') < (op2.user_id or ''):
                op2_prime = EditOperation(
                    type=op2.type,
                    position=op2.position + len(op1.content or ''),
                    length=op2.length,
                    content=op2.content,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp
                )
                return op1, op2_prime
            else:
                op1_prime = EditOperation(
                    type=op1.type,
                    position=op1.position + len(op2.content or ''),
                    length=op1.length,
                    content=op1.content,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp
                )
                return op1_prime, op2
    
    def _transform_insert_delete(
        self,
        insert_op: EditOperation,
        delete_op: EditOperation
    ) -> Tuple[EditOperation, EditOperation]:
        """Transform insert and delete operations."""
        if insert_op.position <= delete_op.position:
            delete_op_prime = EditOperation(
                type=delete_op.type,
                position=delete_op.position + len(insert_op.content or ''),
                length=delete_op.length,
                content=delete_op.content,
                user_id=delete_op.user_id,
                timestamp=delete_op.timestamp
            )
            return insert_op, delete_op_prime
        elif insert_op.position >= delete_op.position + delete_op.length:
            insert_op_prime = EditOperation(
                type=insert_op.type,
                position=insert_op.position - delete_op.length,
                length=insert_op.length,
                content=insert_op.content,
                user_id=insert_op.user_id,
                timestamp=insert_op.timestamp
            )
            return insert_op_prime, delete_op
        else:
            insert_op_prime = EditOperation(
                type=insert_op.type,
                position=delete_op.position,
                length=insert_op.length,
                content=insert_op.content,
                user_id=insert_op.user_id,
                timestamp=insert_op.timestamp
            )
            return insert_op_prime, delete_op
    
    def _transform_delete_delete(
        self,
        op1: EditOperation,
        op2: EditOperation
    ) -> Tuple[EditOperation, EditOperation]:
        """Transform two concurrent delete operations."""
        if op1.position + op1.length <= op2.position:
            op2_prime = EditOperation(
                type=op2.type,
                position=op2.position - op1.length,
                length=op2.length,
                content=op2.content,
                user_id=op2.user_id,
                timestamp=op2.timestamp
            )
            return op1, op2_prime
        elif op2.position + op2.length <= op1.position:
            op1_prime = EditOperation(
                type=op1.type,
                position=op1.position - op2.length,
                length=op1.length,
                content=op1.content,
                user_id=op1.user_id,
                timestamp=op1.timestamp
            )
            return op1_prime, op2
        else:
            overlap_start = max(op1.position, op2.position)
            overlap_end = min(op1.position + op1.length, op2.position + op2.length)
            overlap = overlap_end - overlap_start
            
            if op1.position <= op2.position:
                op1_prime = op1
                op2_prime = EditOperation(
                    type=op2.type,
                    position=op1.position,
                    length=op2.length - overlap,
                    content=op2.content,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp
                )
                return op1_prime, op2_prime
            else:
                op2_prime = op2
                op1_prime = EditOperation(
                    type=op1.type,
                    position=op2.position,
                    length=op1.length - overlap,
                    content=op1.content,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp
                )
                return op1_prime, op2_prime
    
    def _transform_with_replace(
        self,
        op1: EditOperation,
        op2: EditOperation
    ) -> Tuple[EditOperation, EditOperation]:
        """Transform operations involving replace."""
        if op1.type == OperationType.REPLACE:
            delete_op = EditOperation(
                type=OperationType.DELETE,
                position=op1.position,
                length=op1.length,
                user_id=op1.user_id,
                timestamp=op1.timestamp
            )
            insert_op = EditOperation(
                type=OperationType.INSERT,
                position=op1.position,
                length=len(op1.content or ''),
                content=op1.content,
                user_id=op1.user_id,
                timestamp=op1.timestamp
            )
            delete_prime, op2_prime = self.transform_operations(delete_op, op2)
            insert_prime, op2_double_prime = self.transform_operations(insert_op, op2_prime)
            
            op1_prime = EditOperation(
                type=OperationType.REPLACE,
                position=delete_prime.position,
                length=delete_prime.length,
                content=insert_prime.content,
                user_id=op1.user_id,
                timestamp=op1.timestamp
            )
            return op1_prime, op2_double_prime
        else:
            op2_prime, op1_prime = self._transform_with_replace(op2, op1)
            return op1_prime, op2_prime
    
    def apply_operation(self, text: str, operation: EditOperation) -> str:
        """
        Apply an operation to text.
        
        Parameters
        ----------
        text : str
            Original text
        operation : EditOperation
            Operation to apply
            
        Returns
        -------
        str
            Modified text
        """
        if operation.type == OperationType.INSERT:
            return text[:operation.position] + (operation.content or '') + text[operation.position:]
        elif operation.type == OperationType.DELETE:
            return text[:operation.position] + text[operation.position + operation.length:]
        elif operation.type == OperationType.REPLACE:
            return text[:operation.position] + (operation.content or '') + text[operation.position + operation.length:]
        
        return text
    
    def three_way_merge(
        self,
        base: str,
        local: str,
        remote: str
    ) -> Tuple[str, List[Dict]]:
        """
        Perform three-way merge of text content.
        
        Parameters
        ----------
        base : str
            Base version
        local : str
            Local version
        remote : str
            Remote version
            
        Returns
        -------
        Tuple[str, List[Dict]]
            Merged text and list of conflicts
        """
        base_lines = base.splitlines(keepends=True)
        local_lines = local.splitlines(keepends=True)
        remote_lines = remote.splitlines(keepends=True)
        
        merger = difflib.SequenceMatcher(None, base_lines, local_lines)
        local_changes = list(merger.get_opcodes())
        
        merger = difflib.SequenceMatcher(None, base_lines, remote_lines)
        remote_changes = list(merger.get_opcodes())
        
        result = []
        conflicts = []
        i = 0
        
        while i < len(base_lines):
            local_change = self._find_change_at_position(local_changes, i)
            remote_change = self._find_change_at_position(remote_changes, i)
            
            if local_change and remote_change:
                if self._changes_conflict(local_change, remote_change):
                    conflicts.append({
                        'line': i,
                        'local': ''.join(local_lines[local_change[3]:local_change[4]]),
                        'remote': ''.join(remote_lines[remote_change[3]:remote_change[4]])
                    })
                    result.append(f"<<<<<<< LOCAL\n")
                    result.extend(local_lines[local_change[3]:local_change[4]])
                    result.append(f"=======\n")
                    result.extend(remote_lines[remote_change[3]:remote_change[4]])
                    result.append(f">>>>>>> REMOTE\n")
                    i = max(local_change[2], remote_change[2])
                else:
                    if local_change[0] != 'equal':
                        result.extend(local_lines[local_change[3]:local_change[4]])
                        i = local_change[2]
                    elif remote_change[0] != 'equal':
                        result.extend(remote_lines[remote_change[3]:remote_change[4]])
                        i = remote_change[2]
            elif local_change and local_change[0] != 'equal':
                result.extend(local_lines[local_change[3]:local_change[4]])
                i = local_change[2]
            elif remote_change and remote_change[0] != 'equal':
                result.extend(remote_lines[remote_change[3]:remote_change[4]])
                i = remote_change[2]
            else:
                result.append(base_lines[i])
                i += 1
        
        return ''.join(result), conflicts
    
    def _find_change_at_position(self, changes: List, pos: int) -> Optional[Tuple]:
        """Find change that affects a given position."""
        for change in changes:
            if change[1] <= pos < change[2]:
                return change
        return None
    
    def _changes_conflict(self, change1: Tuple, change2: Tuple) -> bool:
        """Check if two changes conflict."""
        return change1[0] != 'equal' and change2[0] != 'equal'
    
    def record_operation(self, operation: EditOperation):
        """
        Record an operation in history.
        
        Parameters
        ----------
        operation : EditOperation
            Operation to record
        """
        self.operation_history.append(operation)
    
    def get_operation_history(self) -> List[EditOperation]:
        """
        Get operation history.
        
        Returns
        -------
        List[EditOperation]
            List of recorded operations
        """
        return self.operation_history.copy()
