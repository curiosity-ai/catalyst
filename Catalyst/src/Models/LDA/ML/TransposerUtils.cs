using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace Catalyst.Models.LDA
{
    internal static class TransposerUtils
    {
        /// <summary>
        /// This is a convenience method that extracts a single slot value's vector,
        /// while simultaneously verifying that there is exactly one value.
        /// </summary>
        public static void GetSingleSlotValue<T>(this ITransposeDataView view, int col, ref VBuffer<T> dst)
        {
            Contracts.CheckValue(view, nameof(view));
            Contracts.CheckParam(0 <= col && col < view.Schema.Count, nameof(col));
            using (var cursor = view.GetSlotCursor(col))
            {
                var getter = cursor.GetGetter<T>();
                if (!cursor.MoveNext())
                    throw Contracts.Except("Could not get single value on column '{0}' because there are no slots", view.Schema[col].Name);
                getter(ref dst);
                if (cursor.MoveNext())
                    throw Contracts.Except("Could not get single value on column '{0}' because there is more than one slot", view.Schema[col].Name);
            }
        }

        /// <summary>
        /// The <see cref="SlotCursor.GetGetter{TValue}"/> is parameterized by a type that becomes the
        /// type parameter for a <see cref="VBuffer{T}"/>, and this is generally preferable and more
        /// sensible but for various reasons it's often a lot simpler to have a get-getter be over
        /// the actual type returned by the getter, that is, parameterize this by the actual
        /// <see cref="VBuffer{T}"/> type.
        /// </summary>
        /// <typeparam name="TValue">The type, must be a <see cref="VBuffer{T}"/> generic type,
        /// though enforcement of this has to be done only at runtime for practical reasons</typeparam>
        /// <param name="cursor">The cursor to get the getter for</param>
        /// <param name="ctx">The exception contxt</param>
        /// <returns>The value getter</returns>
        public static ValueGetter<TValue> GetGetterWithVectorType<TValue>(this SlotCursor cursor, IExceptionContext ctx = null)
        {
            Contracts.CheckValueOrNull(ctx);
            ctx.CheckValue(cursor, nameof(cursor));
            var type = typeof(TValue);
            if (!type.IsGenericEx(typeof(VBuffer<>)))
                throw ctx.Except("Invalid TValue: '{0}'", typeof(TValue));
            var genTypeArgs = type.GetGenericArguments();
            ctx.Assert(genTypeArgs.Length == 1);

            Func<ValueGetter<VBuffer<int>>> del = cursor.GetGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(genTypeArgs[0]);
            var getter = methodInfo.Invoke(cursor, null) as ValueGetter<TValue>;
            if (getter == null)
                throw ctx.Except("Invalid TValue: '{0}'", typeof(TValue));
            return getter;
        }

        /// <summary>
        /// Given a slot cursor, construct a single-column equivalent row cursor, with the single column
        /// active and having the same type. This is useful to exploit the many utility methods that exist
        /// to handle <see cref="DataViewRowCursor"/> and <see cref="DataViewRow"/> but that know nothing about
        /// <see cref="SlotCursor"/>, without having to rewrite all of them. This is, however, rather
        /// something of a hack; whenever possible or reasonable the slot cursor should be used directly.
        /// The name of this column is always "Waffles".
        /// </summary>
        /// <param name="provider">The channel provider used in creating the wrapping row cursor</param>
        /// <param name="cursor">The slot cursor to wrap</param>
        /// <returns>A row cursor with a single active column with the same type as the slot type</returns>
        public static DataViewRowCursor GetRowCursorShim(IChannelProvider provider, SlotCursor cursor)
        {
            Contracts.CheckValue(provider, nameof(provider));
            provider.CheckValue(cursor, nameof(cursor));

            return Utils.MarshalInvoke(GetRowCursorShimCore<int>, cursor.GetSlotType().ItemType.RawType, provider, cursor);
        }

        private static DataViewRowCursor GetRowCursorShimCore<T>(IChannelProvider provider, SlotCursor cursor)
        {
            return new SlotRowCursorShim<T>(provider, cursor);
        }

        /// <summary>
        /// Presents a single transposed column as a single-column dataview.
        /// </summary>
        public sealed class SlotDataView : IDataView
        {
            private readonly IHost _host;
            private readonly ITransposeDataView _data;
            private readonly int _col;
            private readonly DataViewType _type;

            public DataViewSchema Schema { get; }

            public bool CanShuffle => false;

            public SlotDataView(IHostEnvironment env, ITransposeDataView data, int col)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register("SlotDataView");
                _host.CheckValue(data, nameof(data));
                _host.CheckParam(0 <= col && col < data.Schema.Count, nameof(col));
                _type = data.GetSlotType(col);
                _host.AssertValue(_type);

                _data = data;
                _col = col;

                var builder = new DataViewSchema.Builder();
                builder.AddColumn(_data.Schema[_col].Name, _type);
                Schema = builder.ToSchema();
            }

            public long? GetRowCount()
            {
                var type = _data.Schema[_col].Type;
                int valueCount = type.GetValueCount();
                _host.Assert(valueCount > 0);
                return valueCount;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                bool hasZero = columnsNeeded != null && columnsNeeded.Any(x => x.Index == 0);
                return Utils.MarshalInvoke(GetRowCursor<int>, _type.GetItemType().RawType, hasZero);
            }

            private DataViewRowCursor GetRowCursor<T>(bool active)
            {
                return new Cursor<T>(this, active);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
            }

            private sealed class Cursor<T> : RootCursorBase
            {
                private readonly SlotDataView _parent;
                private readonly SlotCursor _slotCursor;
                private readonly Delegate _getter;

                public override DataViewSchema Schema => _parent.Schema;

                public override long Batch => 0;

                public Cursor(SlotDataView parent, bool active)
                    : base(parent._host)
                {
                    _parent = parent;
                    _slotCursor = _parent._data.GetSlotCursor(parent._col);
                    if (active)
                        _getter = _slotCursor.GetGetter<T>();
                }

                /// <summary>
                /// Returns whether the given column is active in this row.
                /// </summary>
                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index == 0, nameof(column));
                    return _getter != null;
                }

                /// <summary>
                /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
                /// This throws if the column is not active in this row, or if the type
                /// <typeparamref name="TValue"/> differs from this column's type.
                /// </summary>
                /// <typeparam name="TValue"> is the column's content type.</typeparam>
                /// <param name="column"> is the output column whose getter should be returned.</param>
                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index == 0, nameof(column));
                    Ch.CheckParam(_getter != null, nameof(column), "requested column not active");

                    var getter = _getter as ValueGetter<TValue>;
                    if (getter == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                    return getter;
                }

                public override ValueGetter<DataViewRowId> GetIdGetter() => GetId;

                private void GetId(ref DataViewRowId id)
                {
                    Ch.Check(_slotCursor.SlotIndex >= 0, RowCursorUtils.FetchValueStateError);
                    id = new DataViewRowId((ulong)_slotCursor.SlotIndex, 0);
                }

                protected override bool MoveNextCore() => _slotCursor.MoveNext();
            }
        }

        // REVIEW: This shim class is very similar to the above shim class, except at the
        // cursor level, not the cursorable level. Is there some non-horrifying way to unify both, somehow?
        private sealed class SlotRowCursorShim<T> : RootCursorBase
        {
            private readonly SlotCursor _slotCursor;

            public override DataViewSchema Schema { get; }

            public override long Batch => 0;

            public SlotRowCursorShim(IChannelProvider provider, SlotCursor cursor)
                : base(provider)
            {
                Contracts.AssertValue(cursor);

                _slotCursor = cursor;
                var builder = new DataViewSchema.Builder();
                builder.AddColumn("Waffles", cursor.GetSlotType());
                Schema = builder.ToSchema();
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.CheckParam(column.Index == 0, nameof(column));
                return true;
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.CheckParam(column.Index == 0, nameof(column));
                return _slotCursor.GetGetterWithVectorType<TValue>(Ch);
            }

            public override ValueGetter<DataViewRowId> GetIdGetter() => GetId;

            private void GetId(ref DataViewRowId id)
            {
                Ch.Check(_slotCursor.SlotIndex >= 0, RowCursorUtils.FetchValueStateError);
                id = new DataViewRowId((ulong)_slotCursor.SlotIndex, 0);
            }

            protected override bool MoveNextCore() => _slotCursor.MoveNext();
        }
    }
}
