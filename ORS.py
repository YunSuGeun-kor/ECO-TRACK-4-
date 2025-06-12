import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
import math
from datetime import datetime, timedelta
import openrouteservice as ors
import openai
import warnings
warnings.filterwarnings('ignore')
from io import StringIO
import polyline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 페이지 설정
st.set_page_config(
    page_title="광양제철소 폐기물 수거 경로 최적화",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WasteRouteOptimizer:
    def __init__(self):
        # ORS API 설정
        self.ors_api_key = "5b3ce3597851110001cf62489c2d4dea20f4405f9f1d318f1e2733c1"
        self.ors_client = ors.Client(key=self.ors_api_key)
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key="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")
        
        self.init_session()
        
    def init_session(self):
        """세션 상태 초기화"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'box_input' not in st.session_state:
            st.session_state.box_input = ""
        if 'optimized_routes' not in st.session_state:
            st.session_state.optimized_routes = None
        if 'route_analysis' not in st.session_state:
            st.session_state.route_analysis = None

    @st.cache_data
    def load_and_process_data(_self, uploaded_file):
        """데이터 로드 및 전처리 (캐싱 적용)"""
        try:
            # 파일 형식에 따른 데이터 로드
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                df = pd.read_excel(uploaded_file)
            
            # 컬럼 정규화
            df = _self.normalize_columns(df)
            
            # 우선순위 계산
            df = _self.calculate_priority_score(df)
            
            return df
            
        except Exception as e:
            st.error(f"파일 로드 오류: {str(e)}")
            return None

    def normalize_columns(self, df):
        """컬럼명 정규화 및 표준화"""
        # 1. 컬럼명 앞뒤 공백 제거
        df.columns = df.columns.str.strip()
        # 컬럼 매핑 (내장 CSV 컬럼명 반영)
        COL_MAP = {
            '박스 위치': '위치',
            '박스 위치 ': '위치',
            '위치명': '위치',
            '위치': '위치',
            '톤': '톤수',
            '부서명': '부서',
            '부서코드': '부서코드',
            '박스구분': '용도',
            '구분': '용도',
            '위도(DD)': '좌표_위도',
            '경도(DD)': '좌표_경도',
            '위도': '좌표_위도',
            '경도': '좌표_경도',
            '위도(DMS)': '위도_DMS',
            '경도(DMS)': '경도_DMS',
            '위치정보(DD)': '위치정보_DD',
            '위치정보(DMS)': '위치정보_DMS'
        }
        # 컬럼명 변경
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        # '위치' 컬럼이 없고 '박스 위치' 또는 '박스 위치 '가 있으면 우선적으로 복사
        if '위치' not in df.columns:
            if '박스 위치' in df.columns:
                df['위치'] = df['박스 위치']
            elif '박스 위치 ' in df.columns:
                df['위치'] = df['박스 위치 ']
        # Unnamed 컬럼 제거
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        # 좌표 데이터 처리
        if '좌표_경도' not in df.columns and '경도(DD)' in df.columns:
            df['좌표_경도'] = pd.to_numeric(df['경도(DD)'], errors='coerce')
        if '좌표_위도' not in df.columns and '위도(DD)' in df.columns:
            df['좌표_위도'] = pd.to_numeric(df['위도(DD)'], errors='coerce')
        if '좌표_경도' in df.columns:
            df['좌표_경도'] = pd.to_numeric(df['좌표_경도'], errors='coerce')
            df['좌표_경도'] = df['좌표_경도'].apply(lambda x: round(x, 10) if pd.notna(x) else x)
        if '좌표_위도' in df.columns:
            df['좌표_위도'] = pd.to_numeric(df['좌표_위도'], errors='coerce')
            df['좌표_위도'] = df['좌표_위도'].apply(lambda x: round(x, 10) if pd.notna(x) else x)
        # 표준 컬럼 정의
        STANDARD_COLS = ['박스번호', '위치', '부서', '좌표_경도', '좌표_위도', '수거빈도', '지연일수', '톤수', '용도', '접수일']
        # 누락된 컬럼 생성
        for col in STANDARD_COLS:
            if col not in df.columns:
                if col in ['수거빈도', '지연일수']:
                    df[col] = np.random.randint(1, 10, len(df))
                elif col == '톤수':
                    # 내장 CSV의 '톤' 컬럼을 '톤수'로 매핑
                    if '톤' in df.columns:
                        df['톤수'] = pd.to_numeric(df['톤'], errors='coerce')
                    else:
                        df[col] = np.random.uniform(0.5, 3.0, len(df))
                elif col == '접수일':
                    df[col] = pd.date_range(start='2024-01-01', periods=len(df))
                elif col == '박스번호':
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = ''
        return df[STANDARD_COLS]

    def calculate_priority_score(self, df):
        """우선순위 점수 계산"""
        # 우선순위 공식: 지연일수 × 수거빈도
        df['priority_score'] = df['지연일수'] * df['수거빈도']
        
        # 분위수 기반 우선순위 등급 (20-60-20)
        q20 = df['priority_score'].quantile(0.2)
        q80 = df['priority_score'].quantile(0.8)
        
        df['우선순위'] = np.select(
            [df['priority_score'] <= q20, df['priority_score'] <= q80],
            [1, 2], 
            default=3
        )
        
        return df

    def calculate_distance_matrix(self, coordinates):
        """거리 행렬 계산 (Haversine 공식 사용)"""
        coords_rad = np.radians(coordinates)
        distances = haversine_distances(coords_rad) * 6371  # 지구 반지름 (km)
        return distances

    def optimize_routes_ors(self, df, num_vehicles=3, vehicle_capacities=None, vehicle_max_counts=None):
        """ORS 기반 차량 경로 최적화"""
        if vehicle_capacities is None:
            vehicle_capacities = [8.5] * num_vehicles
        if vehicle_max_counts is None:
            vehicle_max_counts = [20] * num_vehicles
        if df.empty or '좌표_위도' not in df.columns or '좌표_경도' not in df.columns:
            return None
        
        # 좌표가 유효한 데이터만 필터링 및 문제 수거함 제외
        valid_coords = df.dropna(subset=['좌표_위도', '좌표_경도'])
        # 33번 수거함 제외 (좌표 이상)
        if '박스번호' in valid_coords.columns:
            valid_coords = valid_coords[valid_coords['박스번호'] != 33]
        
        if len(valid_coords) < 2:
            return None
        
        # 차량 수를 1-3대로 제한
        num_vehicles = min(num_vehicles, 3)
        vehicle_capacities = vehicle_capacities[:num_vehicles]
        vehicle_max_counts = vehicle_max_counts[:num_vehicles]
        
        try:
            # ORS 최적화 데이터 준비
            optimization_data = self.prepare_ors_optimization_data(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)

            result = self.ors_client.optimization(jobs=optimization_data['jobs'], vehicles=optimization_data['vehicles'], geometry=True)
            
            if 'routes' in result:
                return self.process_ors_result(result, valid_coords, vehicle_capacities, vehicle_max_counts)
            else:
                st.info("ORS 최적화 결과가 없어 클러스터링 최적화를 사용합니다.")
                return self.optimize_routes_fallback(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)
                
        except Exception as e:
            st.info(f"ORS API 연결 실패. 클러스터링 최적화를 사용합니다. 오류: {str(e)}")
            return self.optimize_routes_fallback(valid_coords, num_vehicles, vehicle_capacities, vehicle_max_counts)

    def prepare_ors_optimization_data(self, df, num_vehicles, vehicle_capacities, vehicle_max_counts):
        """openrouteservice-py 라이브러리용 최적화 데이터 준비"""
        # 가상 시작점 (광양제철소 소각로 위치)
        start_location = [127.765076, 34.926157]  # 경도, 위도 순서
        
        # Jobs 생성 (수거 장소)
        jobs = []
        job_id = 1  # 1부터 시작하는 연속적인 ID
        
        for idx, row in df.iterrows():
            skill_value = int(row['톤수'] * 10)  # 5톤 -> 50, 8.5톤 -> 85

            job = ors.optimization.Job(
                id=job_id,
                location=[row['좌표_경도'], row['좌표_위도']],
                amount=[1],  # 수거량은 1로 고정
                skills=[skill_value],  # 톤수별 스킬
                priority=int(row.get('우선순위', 1))
            )
            jobs.append(job)
            job_id += 1
        
        # Vehicles 생성 - 모든 차량 포함
        vehicles = []
        for i in range(num_vehicles):
            skill_value = int(vehicle_capacities[i] * 10)  # 5톤 -> 50, 8.5톤 -> 85
            
            vehicle = ors.optimization.Vehicle(
                id=i,
                start=start_location,
                end=start_location,
                capacity=[vehicle_max_counts[i]],  # 최대 수거 개수
                skills=[skill_value],  # 차량 톤수별 스킬
                time_window=[0, 28800]
            )
            vehicles.append(vehicle)
        
        # openrouteservice-py 라이브러리용 데이터 형식
        optimization_data = {
            "jobs": jobs,
            "vehicles": vehicles,
        }
        
        # Job ID와 박스번호 매핑을 세션에 저장
        job_to_box_mapping = {}
        job_id = 1
        for idx, row in df.iterrows():
            job_to_box_mapping[job_id] = int(row['박스번호'])
            job_id += 1
        st.session_state.job_to_box_mapping = job_to_box_mapping
        
        return optimization_data

    def process_ors_result(self, ors_result, df, vehicle_capacities, vehicle_max_counts):
        """openrouteservice-py 라이브러리 결과 처리"""
        routes = []
        total_distance = 0
        
        if 'routes' not in ors_result:
            return None
        
        # 세션에서 Job ID와 박스번호 매핑 가져오기
        job_to_box_mapping = getattr(st.session_state, 'job_to_box_mapping', {})
        
        # 모든 설정된 차량에 대해 경로 생성
        for i in range(len(vehicle_capacities)):
            vehicle_capacity = vehicle_capacities[i]
            max_count = vehicle_max_counts[i]
            
            # 해당 차량의 경로 찾기
            route_info = None
            for route in ors_result['routes']:
                if route['vehicle'] == i:
                    route_info = route
                    break
            
            if route_info and 'steps' in route_info:
                # 경로에 포함된 작업(수거 장소) 추출
                job_ids = [step['job'] for step in route_info['steps'] if 'job' in step]
                
                # Job ID를 박스번호로 변환
                box_numbers = [job_to_box_mapping.get(job_id) for job_id in job_ids if job_id in job_to_box_mapping]
                box_numbers = [box_num for box_num in box_numbers if box_num is not None]
                
                # 해당 작업들의 데이터 가져오기
                if box_numbers:
                    route_data = df[df['박스번호'].isin(box_numbers)].copy()
                else:
                    route_data = pd.DataFrame()
                
                # 경로 거리 계산
                route_distance = route_info.get('distance', 0) / 1000.0 if 'distance' in route_info else 0
                total_distance += route_distance
                
                # 지오메트리 정보 추출
                geometry = route_info.get('geometry', None)
                
            else:
                # 경로가 할당되지 않은 차량
                route_data = pd.DataFrame()
                route_distance = 0
                geometry = None
            
            routes.append({
                'vehicle_id': i + 1,
                'data': route_data,
                'distance': route_distance,
                'total_tonnage': route_data['톤수'].sum() if not route_data.empty else 0,
                'capacity': vehicle_capacity,
                'collection_count': len(route_data),
                'max_count': max_count,
                'tonnage_type': f"{vehicle_capacity}톤 전용",
                'high_priority_count': len(route_data[route_data['우선순위'] >= 2]) if not route_data.empty else 0,
                'geometry': geometry,
                'ors_optimized': True,
                'has_assignments': not route_data.empty
            })
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_vehicles': len(routes),
            'active_vehicles': len([r for r in routes if r['has_assignments']]),
            'optimization_method': 'ORS (OpenRouteService) Python Library',
            'ors_result': ors_result
        }

    def optimize_routes_fallback(self, df, num_vehicles, vehicle_capacities, vehicle_max_counts):
        """ORS 실패 시 대체 클러스터링 방법 - 모든 차량 활용"""
        routes = []
        total_distance = 0
        
        # 톤수별로 데이터 그룹화
        tonnage_groups = {}
        for tonnage in df['톤수'].unique():
            tonnage_groups[tonnage] = df[df['톤수'] == tonnage].copy().reset_index(drop=True)
        
        # 모든 차량에 대해 경로 생성
        for i in range(num_vehicles):
            vehicle_capacity = vehicle_capacities[i]
            max_count = vehicle_max_counts[i]
            
            # 해당 톤수의 데이터가 있는지 확인
            if vehicle_capacity in tonnage_groups and not tonnage_groups[vehicle_capacity].empty:
                # 우선순위 기반 정렬
                tonnage_data = tonnage_groups[vehicle_capacity].sort_values(
                    ['우선순위', 'priority_score'], 
                    ascending=[False, False]
                )
                
                # 차량당 할당할 데이터 계산
                vehicles_for_tonnage = sum(1 for cap in vehicle_capacities if cap == vehicle_capacity)
                data_per_vehicle = len(tonnage_data) // vehicles_for_tonnage
                remainder = len(tonnage_data) % vehicles_for_tonnage
                
                # 현재 차량이 해당 톤수의 몇 번째 차량인지 계산
                vehicle_order = sum(1 for j in range(i) if vehicle_capacities[j] == vehicle_capacity)
                
                # 할당할 데이터 범위 계산
                start_idx = vehicle_order * data_per_vehicle
                end_idx = start_idx + data_per_vehicle
                
                # 나머지 데이터를 첫 번째 차량들에 배분
                if vehicle_order < remainder:
                    start_idx += vehicle_order
                    end_idx += vehicle_order + 1
                else:
                    start_idx += remainder
                    end_idx += remainder
                
                # 최대 수거 개수 제한 적용
                assigned_data = tonnage_data.iloc[start_idx:min(end_idx, start_idx + max_count)].copy()
                
                if not assigned_data.empty:
                    # 간단한 거리 계산
                    route_distance = self.calculate_route_distance(assigned_data[['좌표_위도', '좌표_경도']].values)
                    total_distance += route_distance
                else:
                    assigned_data = pd.DataFrame()
                    route_distance = 0
            else:
                # 해당 톤수의 데이터가 없는 경우 빈 경로
                assigned_data = pd.DataFrame()
                route_distance = 0
            
            routes.append({
                'vehicle_id': i + 1,
                'data': assigned_data,
                'distance': route_distance,
                'total_tonnage': assigned_data['톤수'].sum() if not assigned_data.empty else 0,
                'capacity': vehicle_capacity,
                'collection_count': len(assigned_data),
                'max_count': max_count,
                'tonnage_type': f"{vehicle_capacity}톤 전용",
                'high_priority_count': len(assigned_data[assigned_data['우선순위'] >= 2]) if not assigned_data.empty else 0,
                'ors_optimized': False,
                'has_assignments': not assigned_data.empty
            })
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_vehicles': len(routes),
            'active_vehicles': len([r for r in routes if r['has_assignments']]),
            'optimization_method': '대체 클러스터링 최적화 (전체 차량 활용)'
        }

    def nearest_neighbor_tsp(self, coordinates):
        """최근접 이웃 TSP 휴리스틱"""
        if len(coordinates) <= 1:
            return list(range(len(coordinates)))
        
        n = len(coordinates)
        distance_matrix = self.calculate_distance_matrix(coordinates)
        
        unvisited = set(range(1, n))
        route = [0]  # 시작점
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return route

    def calculate_route_distance(self, coordinates):
        """경로 총 거리 계산"""
        if len(coordinates) <= 1:
            return 0
        
        total_distance = 0
        for i in range(len(coordinates) - 1):
            coord1 = np.radians([coordinates[i]])
            coord2 = np.radians([coordinates[i + 1]])
            distance = haversine_distances(coord1, coord2)[0][0] * 6371
            total_distance += distance
        
        return total_distance

    def create_route_map(self, routes_data):
        """경로 지도 시각화 (Folium 기반, 차량별 토글 가능, 박스번호 중앙 표시)"""
        if not routes_data or not routes_data['routes']:
            return None

        # 진한 색상 팔레트 (차량별)
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f']

        # 중심 좌표 계산
        all_lats = []
        all_lons = []
        for route in routes_data['routes']:
            if not route['data'].empty:
                all_lats.extend(route['data']['좌표_위도'].tolist())
                all_lons.extend(route['data']['좌표_경도'].tolist())
        if not all_lats:
            return None
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

        # Folium 지도 생성
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='OpenStreetMap'
        )

        # 광양제철소 시작점 표시 (고정)
        start_location = [34.926157, 127.765076]
        folium.Marker(
            location=start_location,
            popup='<b>광양제철소</b><br>차량 출발/도착지',
            tooltip='광양제철소 (시작점)',
            icon=folium.Icon(color='black', icon='home', prefix='glyphicon')
        ).add_to(m)

        # 각 차량의 경로/마커를 FeatureGroup으로 분리
        for i, route in enumerate(routes_data['routes']):
            route_df = route['data']
            if route_df.empty:
                continue
            color = colors[i % len(colors)]
            vehicle_id = route["vehicle_id"]
            # 차량 유형 표시 (5톤, 8.5톤)
            vehicle_type = f"{route['capacity']}톤"
            # HTML 색상 스와치와 차량 유형 포함
            group_name = f'<span style="display:inline-block;width:14px;height:14px;background:{color};border-radius:50%;margin-right:6px;vertical-align:middle;"></span> 차량 {vehicle_id} ({vehicle_type})'
            vehicle_group = folium.FeatureGroup(name=group_name, show=True)
            # 수거함 위치 CircleMarker + 박스번호 중앙 표시
            for idx, row in route_df.iterrows():
                folium.CircleMarker(
                    location=[row['좌표_위도'], row['좌표_경도']],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.95,
                    popup=folium.Popup(f"<b>박스 {row['박스번호']}</b><br>위치: {row['위치']}<br>부서: {row['부서']}<br>톤수: {row['톤수']}<br>용도: {row['용도']}", max_width=250),
                    tooltip=f"박스 {row['박스번호']} - {row['위치']}"
                ).add_to(vehicle_group)
                # 박스번호를 원 중앙에 표시 (DivIcon)
                folium.map.Marker(
                    [row['좌표_위도'], row['좌표_경도']],
                    icon=folium.DivIcon(
                        html=f'<div style="display:flex;align-items:center;justify-content:center;width:14px;height:14px;font-size:11px;font-weight:bold;color:white;text-align:center;pointer-events:none;">{row["박스번호"]}</div>',
                        icon_size=(14, 14),
                        icon_anchor=(7, 7)
                    )
                ).add_to(vehicle_group)
            # ORS geometry(실제 도로 경로) 표시
            geometry = route.get('geometry')
            coords = None
            if geometry:
                if isinstance(geometry, str):
                    try:
                        coords = polyline.decode(geometry)
                        coords = [[lat, lon] for lat, lon in coords]
                    except Exception:
                        coords = None
                elif isinstance(geometry, dict) and 'coordinates' in geometry:
                    coords = [[lat, lon] for lon, lat in geometry['coordinates']]
                elif isinstance(geometry, list) and len(geometry) > 0:
                    if all(isinstance(coord, (list, tuple)) and len(coord) >= 2 for coord in geometry):
                        coords = [[lat, lon] for lon, lat in geometry]
            if coords and len(coords) > 1:
                folium.PolyLine(
                    locations=coords,
                    weight=4,
                    color=color,
                    opacity=0.85,
                    popup=f'차량 {vehicle_id} 실제 경로'
                ).add_to(vehicle_group)
            elif len(route_df) > 1:
                coords = [[row['좌표_위도'], row['좌표_경도']] for _, row in route_df.iterrows()]
                folium.PolyLine(
                    locations=coords,
                    weight=4,
                    color=color,
                    opacity=0.85,
                    popup=f'차량 {vehicle_id} 경로'
                ).add_to(vehicle_group)
            vehicle_group.add_to(m)

        # 차량별 FeatureGroup을 LayerControl로 토글 (HTML 허용)
        folium.LayerControl(collapsed=False, position='topright',
            # HTML legend labels
            ).add_to(m)
        # 기존 범례 제거 (LayerControl이 범례 역할)
        return m

    def decode_polyline(self, polyline_str):
        """Polyline 문자열을 좌표 배열로 디코딩"""
        try:
            # 간단한 polyline 디코딩 (실제 구현은 더 복잡할 수 있음)
            # 여기서는 기본적인 처리만 수행
            return []
        except:
            return []

    def create_dashboard_charts(self, df):
        """대시보드 차트 생성"""
        charts = {}
        
        # 1. 부서별 분포 (막대 차트)
        dept_counts = df['부서'].value_counts().head(10)
        dept_df = pd.DataFrame({
            '부서': dept_counts.index,
            '수거함_수': dept_counts.values
        })
        charts['dept_distribution'] = px.bar(
            dept_df,
            x='수거함_수',
            y='부서',
            orientation='h',
            title="부서별 수거함 분포 (상위 10개)",
            labels={'수거함_수': '수거함 수', '부서': '부서'}
        ).update_layout(yaxis={'categoryorder':'total ascending'})
        
        # 2. 우선순위 분포 (파이 차트)
        priority_counts = df['우선순위'].value_counts().sort_index()
        priority_labels = {1: '낮음(1)', 2: '보통(2)', 3: '높음(3)'}
        charts['priority_distribution'] = px.pie(
            values=priority_counts.values,
            names=[priority_labels[i] for i in priority_counts.index],
            title="우선순위별 분포",
            color_discrete_sequence=['green', 'orange', 'red']
        )
        
        # 3. 톤수 분포 (막대그래프, 5톤/8.5톤만, 색상 다르게, 간격 없음, 얇은 막대)
        tonnage_counts = df['톤수'].value_counts().sort_index()
        tonnage_df = pd.DataFrame({
            '톤수': tonnage_counts.index.astype(str) + '톤',
            '수거함_수': tonnage_counts.values
        })
        charts['tonnage_distribution'] = px.bar(
            tonnage_df,
            x='톤수',
            y='수거함_수',
            color='톤수',
            color_discrete_sequence=['#1f77b4', '#ff7f0e'],
            title="톤수 분포",
            labels={'톤수': '톤수', '수거함_수': '수거함 수'},
        ).update_traces(width=0.4).update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':['5.0톤','8.5톤']},
            bargap=0,
            showlegend=False
        )
        
        # 4. 용도별 분포 (도넛 차트)
        usage_counts = df['용도'].value_counts()
        charts['usage_distribution'] = px.pie(
            values=usage_counts.values,
            names=usage_counts.index,
            title="용도별 수거함 분포",
            hole=0.4
        )
        
        return charts

    def get_vehicle_assignment(self, box_number, routes_data):
        """박스번호에 해당하는 수거 차량 번호 반환"""
        if not routes_data or 'routes' not in routes_data:
            return '미배정'
        
        for route in routes_data['routes']:
            if box_number in route['data']['박스번호'].values:
                return f"차량 {route['vehicle_id']}"
        
        return '미배정'

    def display_metrics(self, df, routes_data=None):
        """주요 메트릭 표시"""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 수거함 수", len(df))
        with col2:
            if routes_data:
                active_vehicles = routes_data.get('active_vehicles', routes_data['total_vehicles'])
                total_vehicles = routes_data['total_vehicles']
                st.metric("차량 활용률", f"{active_vehicles}/{total_vehicles}대")
        col3, col4, col5 = st.columns(3)
        with col3:
            incineration_count = len(df[df['용도'].str.contains('소각', na=False)])
            st.metric("소각용 수거함", incineration_count)
        with col4:
            recycling_count = len(df[df['용도'].str.contains('재활용', na=False)])
            st.metric("재활용 수거함", recycling_count)
        with col5:
            if routes_data:
                st.metric("총 이동거리", f"{routes_data['total_distance']:.1f}km")

    def display_route_analysis(self, routes_data):
        """경로 분석 결과 표시"""
        if not routes_data:
            return
        
        st.subheader("🚛 경로 최적화 분석")
        
        # 최적화 방법 표시
        optimization_method = routes_data.get('optimization_method', '알 수 없음')
        if 'ORS' in optimization_method:
            st.success(f"✅ {optimization_method}")
            st.info("OpenRouteService를 활용한 전문적인 차량 경로 최적화가 적용되었습니다.")
        else:
            st.info(f"📊 {optimization_method}")
        
        # 클러스터링 효율성 요약 (ORS가 아닌 경우에만)
        if 'cluster_efficiency' in routes_data and 'ORS' not in optimization_method:
            st.write("**지리적 클러스터링 효율성**")
            efficiency_cols = st.columns(len(routes_data['cluster_efficiency']))
            
            for i, (tonnage, stats) in enumerate(routes_data['cluster_efficiency'].items()):
                with efficiency_cols[i]:
                    avg_distance = stats['total_distance'] / stats['total_routes'] if stats['total_routes'] > 0 else 0
                    avg_collections = stats['total_collections'] / stats['total_routes'] if stats['total_routes'] > 0 else 0
                    
                    st.metric(
                        f"{tonnage}톤 차량", 
                        f"{stats['total_routes']}대",
                        f"평균 {avg_distance:.1f}km, {avg_collections:.0f}개"
                    )
        
        # 경로별 상세 정보
        for route in routes_data['routes']:
            has_assignments = route.get('has_assignments', len(route['data']) > 0)
            collection_count = route.get('collection_count', 0)
            max_count = route.get('max_count', 0)
            
            # 차량 상태 표시
            status_icon = "🚛" if has_assignments else "🚚"
            status_text = "활성" if has_assignments else "대기"
            
            tonnage_info = f" | {route['tonnage_type']} ({collection_count}/{max_count}개) - {status_text}"
            
            with st.expander(f"{status_icon} 차량 {route['vehicle_id']} - {collection_count}개소 ({route['distance']:.1f}km){tonnage_info}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("이동거리", f"{route['distance']:.1f}km")
                
                with col2:
                    st.metric("수거 개수", f"{collection_count}/{max_count}개")
                
                with col3:
                    st.metric("수거 유형", route['tonnage_type'])
                
                with col4:
                    st.metric("고우선순위", f"{route['high_priority_count']}개")
                
                # 경로 상세 테이블 (할당된 작업이 있는 경우만)
                if has_assignments and not route['data'].empty:
                    st.dataframe(
                        route['data'][['박스번호', '위치', '부서', '우선순위', '톤수', '용도']],
                        use_container_width=True
                    )
                elif not has_assignments:
                    st.info("이 차량에는 할당된 수거 작업이 없습니다.")
                

    def generate_collection_insights(self, df, routes_data):
        """OpenAI를 활용한 수거 인사이트 분석"""
        try:
            # 데이터 요약 생성
            data_summary = {
                "총_수거함_수": len(df),
                "부서_수": df['부서'].nunique(),
                "평균_우선순위": df['우선순위'].mean(),
                "톤수_분포": df['톤수'].value_counts().to_dict(),
                "우선순위_분포": df['우선순위'].value_counts().to_dict(),
                "부서별_분포": df['부서'].value_counts().head(5).to_dict()
            }
            
            # 경로 최적화 결과 요약
            if routes_data:
                route_summary = {
                    "총_차량_수": routes_data['total_vehicles'],
                    "활성_차량_수": routes_data.get('active_vehicles', routes_data['total_vehicles']),
                    "총_거리": round(routes_data['total_distance'], 1),
                    "최적화_방식": routes_data['optimization_method'],
                    "차량별_정보": []
                }
                
                for route in routes_data['routes']:
                    route_summary["차량별_정보"].append({
                        "차량_번호": route['vehicle_id'],
                        "수거_개수": route['collection_count'],
                        "최대_수거량": route['max_count'],
                        "거리": round(route['distance'], 1),
                        "톤수_유형": route['capacity'],
                        "고우선순위_개수": route['high_priority_count']
                    })
            else:
                route_summary = {"message": "경로 최적화 결과가 없습니다."}
            
            # OpenAI 프롬프트 생성
            prompt = f"""
            다음은 광양제철소 폐기물 수거 데이터 분석 결과입니다:

            **데이터 현황:**
            - 총 수거함 수: {data_summary['총_수거함_수']}개
            - 관련 부서 수: {data_summary['부서_수']}개
            - 평균 우선순위: {data_summary['평균_우선순위']:.1f}
            - 톤수 분포: {data_summary['톤수_분포']}
            - 우선순위 분포: {data_summary['우선순위_분포']}
            - 주요 부서별 분포: {data_summary['부서별_분포']}

            **경로 최적화 결과:**
            {route_summary}

            위 데이터를 바탕으로 다음 내용을 포함한 전문적인 수거 인사이트 보고서를 작성해주세요:

            1. **현황 분석**: 현재 수거 시스템의 특징과 패턴
            2. **효율성 평가**: 경로 최적화 결과의 효율성 분석
            3. **개선 제안**: 구체적이고 실행 가능한 최적화 방안
            4. **예상 효과**: 제안사항 적용 시 기대되는 개선 효과

            보고서는 한국어로 작성하고, 제철소 관리자가 이해하기 쉽도록 명확하고 간결하게 작성해주세요.
            """
            
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "당신은 산업용 폐기물 수거 최적화 전문가입니다. 데이터를 분석하여 실용적인 인사이트와 개선방안을 제시합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"인사이트 생성 중 오류가 발생했습니다: {str(e)}"

    def create_vehicle_map(self, route_data):
        """차량별 개별 지도 생성 (경로 포함)"""
        if route_data.empty:
            return None
            
        fig = go.Figure()
        
        # 광양제철소 시작점 표시
        start_location = [34.9006, 127.7669]  # 위도, 경도 순서
        fig.add_trace(go.Scattermapbox(
            lat=[start_location[0]],
            lon=[start_location[1]],
            mode='markers',
            marker=dict(size=20, color='green', symbol='star'),
            name='광양제철소 (출발/도착)',
            hovertemplate='<b>광양제철소</b><br>차량 출발/도착지<extra></extra>'
        ))
        
        # 우선순위별 색상 설정
        priority_colors = {1: 'lightblue', 2: 'orange', 3: 'red'}
        priority_names = {1: '낮음', 2: '보통', 3: '높음'}
        
        # 우선순위별로 수거함 표시
        for priority in sorted(route_data['우선순위'].unique()):
            priority_data = route_data[route_data['우선순위'] == priority]
            if not priority_data.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=priority_data['좌표_위도'],
                    lon=priority_data['좌표_경도'],
                    mode='markers+text',
                    marker=dict(size=14, color=priority_colors[priority]),
                    text=priority_data['박스번호'].astype(str),
                    textposition="top center",
                    textfont=dict(size=10, color='black'),
                    name=f'우선순위 {priority} ({priority_names[priority]})',
                    hovertemplate='<b>박스번호:</b> %{text}<br>' +
                                 '<b>위치:</b> %{customdata[0]}<br>' +
                                 '<b>부서:</b> %{customdata[1]}<br>' +
                                 '<b>우선순위:</b> %{customdata[2]}<br>' +
                                 '<b>톤수:</b> %{customdata[3]:.1f}톤<extra></extra>',
                    customdata=priority_data[['위치', '부서', '우선순위', '톤수']].values
                ))
        
        # 수거 경로 최적화된 순서로 연결
        if len(route_data) >= 1:
            # 우선순위 순으로 정렬하여 경로 생성
            sorted_data = route_data.sort_values(['우선순위', 'priority_score'], ascending=[False, False])
            
            # 시작점 → 첫 번째 수거함
            if len(sorted_data) > 0:
                route_lats = [start_location[0]]
                route_lons = [start_location[1]]
                
                # 수거함들을 우선순위 순으로 연결
                for _, row in sorted_data.iterrows():
                    route_lats.append(row['좌표_위도'])
                    route_lons.append(row['좌표_경도'])
                
                # 마지막 수거함 → 시작점 (복귀)
                route_lats.append(start_location[0])
                route_lons.append(start_location[1])
                
                # 경로선 그리기
                fig.add_trace(go.Scattermapbox(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines',
                    line=dict(width=4, color='blue', dash='solid'),
                    name='최적 수거 경로',
                    hovertemplate='수거 경로<extra></extra>'
                ))
                
                # 방향 표시를 위한 화살표 (선택적)
                for i in range(len(route_lats)-1):
                    mid_lat = (route_lats[i] + route_lats[i+1]) / 2
                    mid_lon = (route_lons[i] + route_lons[i+1]) / 2
                    
                    if i < 3:  # 처음 몇 개만 방향 표시
                        fig.add_trace(go.Scattermapbox(
                            lat=[mid_lat],
                            lon=[mid_lon],
                            mode='markers',
                            marker=dict(size=8, color='darkblue', symbol='triangle-up'),
                            name='방향',
                            showlegend=False,
                            hovertemplate=f'경로 순서: {i+1}<extra></extra>'
                        ))
        
        # 지도 레이아웃 설정
        all_lats = list(route_data['좌표_위도']) + [start_location[0]]
        all_lons = list(route_data['좌표_경도']) + [start_location[1]]
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # 적절한 줌 레벨 계산
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        zoom_level = max(10, min(14, 12 - max(lat_range, lon_range) * 10))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom_level
            ),
            height=500,
            title=f"차량별 수거 경로 및 대상 위치",
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def display_insights_section(self, df, routes_data):
        """인사이트 섹션 표시"""
        st.subheader("🤖 AI 수거 인사이트 분석")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("OpenAI를 활용하여 수거 데이터를 분석하고 최적화 인사이트를 생성합니다.")
        
        with col2:
            if st.button("📊 인사이트 생성", type="primary"):
                with st.spinner("AI가 데이터를 분석 중입니다..."):
                    insights = self.generate_collection_insights(df, routes_data)
                    st.session_state.insights = insights
        
        # 인사이트 표시
        if hasattr(st.session_state, 'insights') and st.session_state.insights:
            st.write("### 📋 분석 결과")
            st.write(st.session_state.insights)

    def run(self):
        """메인 애플리케이션 실행"""
        # 헤더
        st.title("🚛 광양제철소 폐기물 수거 경로 최적화")
        st.markdown("AI 기반 최적 경로 계획으로 효율적인 폐기물 수거를 실현합니다.")
        
        # 사이드바 - 설정
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "📂 데이터 파일 업로드",
                type=["csv", "xlsx", "xls"],
                help="CSV 또는 Excel 파일을 업로드하세요."
            )
            
            # 경로 최적화 설정
            st.subheader("🚚 차량 설정")
            vehicle_options = [1, 2, 3]
            selected_vehicles = st.multiselect(
                "투입할 차량 선택",
                options=vehicle_options,
                default=[1, 2, 3],
                help="최적화에 사용할 차량을 선택하세요"
            )
            num_vehicles = len(selected_vehicles) if selected_vehicles else 1
            st.caption(f"선택된 차량: {num_vehicles}대")
            
            # 개별 차량 수거 유형 설정
            st.write("**개별 차량 수거 유형 설정**")
            st.caption("각 차량이 수거할 수 있는 톤수 유형과 최대 수거 개수를 선택하세요")
            vehicle_capacities = []
            vehicle_max_counts = []
            
            if selected_vehicles:
                for i, vehicle_num in enumerate(selected_vehicles):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        capacity = st.selectbox(
                            f"차량 {vehicle_num} 수거 유형",
                            options=[5, 8.5],
                            format_func=lambda x: f"{x}톤 수거함 전용",
                            index=1,  # 기본값 8.5톤
                            key=f"vehicle_{vehicle_num}_capacity"
                        )
                        vehicle_capacities.append(capacity)
                    
                    with col2:
                        max_count = st.number_input(
                            f"최대 수거 개수",
                            min_value=1,
                            max_value=50,
                            value=6,
                            key=f"vehicle_{vehicle_num}_max_count",
                            help="해당 차량이 수거할 수 있는 최대 수거함 개수"
                        )
                        vehicle_max_counts.append(max_count)
            else:
                st.warning("차량을 선택해주세요.")
                vehicle_capacities = [8.5]
                vehicle_max_counts = [6]
            
            # 차량별 수거 유형 요약
            capacity_summary = {}
            for i, cap in enumerate(vehicle_capacities):
                if cap not in capacity_summary:
                    capacity_summary[cap] = []
                capacity_summary[cap].append(vehicle_max_counts[i])
            
            summary_parts = []
            for cap, counts in capacity_summary.items():
                avg_count = sum(counts) / len(counts)
                summary_parts.append(f"{cap}톤: {len(counts)}대 (평균 {avg_count:.0f}개)")
            
            summary_text = ", ".join(summary_parts)
            st.info(f"차량 배치: {summary_text}")
            
            # 박스번호 필터 (삭제)
            # st.subheader("📦 박스 필터")
            # box_input = st.text_area(
            #     "특정 박스번호 입력",
            #     value=st.session_state.box_input,
            #     height=100,
            #     help="쉼표, 공백, 줄바꿈으로 구분하여 입력"
            # )
            # if box_input != st.session_state.box_input:
            #     st.session_state.box_input = box_input

        # 데이터 로드
        if uploaded_file:
            df = self.load_and_process_data(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ 데이터 {len(df)}건 로드 완료")
        
        # 메인 컨텐츠
        df = st.session_state.get("data")
        if df is None or df.empty:
            st.info("📁 데이터 파일을 업로드하여 시작하세요.")
            return
        
        # 박스번호 필터링 (삭제)
        # if box_input and box_input.strip():
        #     selected_boxes = re.findall(r'\d+', box_input)
        #     if selected_boxes:
        #         selected_boxes = list(map(int, selected_boxes))
        #         df = df[df['박스번호'].isin(selected_boxes)]
        #         st.info(f"🔍 {len(selected_boxes)}개 박스 선택됨 → 표시된 데이터: {len(df)}개")

        # 경로 최적화 실행
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 데이터 현황")
        with col2:
            if st.button("🚀 경로 최적화 실행", type="primary"):
                if not selected_vehicles:
                    st.error("❌ 차량을 선택해주세요.")
                elif not vehicle_capacities or len(vehicle_capacities) != len(selected_vehicles):
                    st.error("❌ 차량 설정을 완료해주세요.")
                else:
                    with st.spinner("ORS 기반 경로 최적화 진행 중..."):
                        routes_data = self.optimize_routes_ors(df, num_vehicles, vehicle_capacities, vehicle_max_counts)
                        if routes_data:
                            st.session_state.optimized_routes = routes_data
                            optimization_method = routes_data.get('optimization_method', 'ORS 최적화')
                            st.success(f"✅ 경로 최적화 완료! ({optimization_method})")
                        else:
                            st.error("❌ 좌표 데이터가 부족하여 경로 최적화를 실행할 수 없습니다.")
        
        # 메트릭 표시
        routes_data = st.session_state.get('optimized_routes')
        self.display_metrics(df, routes_data)
        
        # 탭 구성
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 대시보드", "🗺️ 경로 지도", "📋 데이터 테이블", "📊 경로 분석", "💬 유저 피드백"])
        
        with tab1:
            st.subheader("📊 데이터 분석 대시보드")
            charts = self.create_dashboard_charts(df)
            # 차트 2x2 레이아웃 (간격 조정)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.plotly_chart(charts['dept_distribution'], use_container_width=True)
                st.plotly_chart(charts['tonnage_distribution'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['priority_distribution'], use_container_width=True)
                st.plotly_chart(charts['usage_distribution'], use_container_width=True)
        
        with tab2:
            st.subheader("🗺️ 최적화된 수거 경로")
            if routes_data:
                route_map = self.create_route_map(routes_data)
                if route_map:
                    st.markdown("""
                    <div style='display: flex; justify-content: center; align-items: center; width: 100%; max-width: 1400px; margin: 0 auto; padding: 0 16px;'>
                    """, unsafe_allow_html=True)
                    st_folium(route_map, width='100%', height=650)
                    st.markdown("""
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ 좌표 데이터가 부족하여 지도를 표시할 수 없습니다.")
            else:
                st.info("🚀 '경로 최적화 실행' 버튼을 클릭하여 최적 경로를 확인하세요.")
        
        with tab3:
            st.subheader("📋 수거함 데이터")
            
            # 데이터 필터링 옵션
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                priority_filter = st.multiselect(
                    "우선순위 필터",
                    options=[1, 2, 3],
                    default=[1, 2, 3]
                )
            
            with col2:
                dept_filter = st.multiselect(
                    "부서 필터",
                    options=df['부서'].unique(),
                    default=df['부서'].unique()
                )
            
            with col3:
                usage_filter = st.multiselect(
                    "용도 필터",
                    options=df['용도'].unique(),
                    default=df['용도'].unique()
                )
            
            with col4:
                # 수거 차량 필터 (경로 최적화 결과가 있는 경우만)
                if routes_data:
                    vehicle_options = ['전체'] + [f'차량 {i}' for i in range(1, routes_data['total_vehicles'] + 1)] + ['미배정']
                    vehicle_filter = st.selectbox(
                        "수거 차량 필터",
                        options=vehicle_options,
                        index=0
                    )
                else:
                    vehicle_filter = '전체'
            
            # 수거 차량 번호 추가 (필터 적용 전에)
            def get_vehicle_for_box(box_number):
                if not routes_data or 'routes' not in routes_data:
                    return '미배정'
                
                for route in routes_data['routes']:
                    if box_number in route['data']['박스번호'].values:
                        return f"차량 {route['vehicle_id']}"
                
                return '미배정'
            
            df_with_vehicle = df.copy()
            if routes_data:
                df_with_vehicle['수거 차량'] = df_with_vehicle['박스번호'].apply(get_vehicle_for_box)
            else:
                df_with_vehicle['수거 차량'] = '미배정'
            
            # 필터 적용
            filtered_df = df_with_vehicle[
                (df_with_vehicle['우선순위'].isin(priority_filter)) &
                (df_with_vehicle['부서'].isin(dept_filter)) &
                (df_with_vehicle['용도'].isin(usage_filter))
            ].copy()
            
            # 차량 필터 적용
            if vehicle_filter != '전체':
                filtered_df = filtered_df[filtered_df['수거 차량'] == vehicle_filter].copy()
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # 다운로드 버튼
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 필터된 데이터 다운로드 (CSV)",
                data=csv,
                file_name=f"filtered_waste_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab4:
            if routes_data:
                self.display_route_analysis(routes_data)
                
                # AI 인사이트 섹션 추가
                st.markdown("---")
                self.display_insights_section(df, routes_data)
            else:
                st.info("🚀 경로 최적화를 먼저 실행해주세요.")
        
        with tab5:
            st.subheader("💬 유저 피드백 보내기")
            st.write("서비스 개선을 위한 의견이나 불편사항, 제안사항을 자유롭게 남겨주세요!")
            with st.form("feedback_form"):
                user_name = st.text_input("이름 (선택)")
                user_email = st.text_input("이메일 (선택)")
                feedback = st.text_area("피드백 내용", max_chars=1000, height=180)
                submitted = st.form_submit_button("피드백 전송")
            if submitted:
                if not feedback.strip():
                    st.error("피드백 내용을 입력해주세요.")
                else:
                    try:
                        # 메일 전송 설정 (SMTP 정보는 실제 운영시 환경변수 등으로 관리 권장)
                        smtp_host = 'smtp.example.com'  # 실제 SMTP 서버 주소로 변경
                        smtp_port = 587
                        smtp_user = 'your_email@example.com'  # 실제 발신자 이메일
                        smtp_pass = 'your_password'           # 실제 비밀번호
                        sender = smtp_user
                        receiver = 'cf100@posco.com'
                        subject = '[광양제철소 폐기물 경로 최적화] 유저 피드백'
                        body = f"""
                        [유저 피드백 도착]
                        이름: {user_name}
                        이메일: {user_email}
                        내용:\n{feedback}
                        """
                        msg = MIMEMultipart()
                        msg['From'] = sender
                        msg['To'] = receiver
                        msg['Subject'] = subject
                        msg.attach(MIMEText(body, 'plain'))
                        with smtplib.SMTP(smtp_host, smtp_port) as server:
                            server.starttls()
                            server.login(smtp_user, smtp_pass)
                            server.sendmail(sender, receiver, msg.as_string())
                        st.success("피드백이 성공적으로 전송되었습니다. 소중한 의견 감사합니다!")
                    except Exception as e:
                        st.error(f"피드백 전송 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 애플리케이션 실행
    app = WasteRouteOptimizer()
    app.run()
