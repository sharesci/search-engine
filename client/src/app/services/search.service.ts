import { Injectable } from '@angular/core'
import { Http, Response, URLSearchParams } from '@angular/http'
import { Observable } from 'rxjs';
import { AppConfig } from '../app.config.js';
import 'rxjs/add/operator/map';

@Injectable()
export class SearchService {
    
    constructor(private _http: Http, private _config: AppConfig) { }

    private _searchUrl = this._config.apiUrl + '/api/v1/search?';

    search(searchToken: string, offset = 0, maxResult = "10"): Observable<any> {
        let queryString = new URLSearchParams();
        queryString.append('any', searchToken);
        queryString.append('offset', offset.toString());
        queryString.append('maxResults', maxResult);        

        return this._http.get(this._searchUrl + queryString.toString())
            .map((response: Response) => response.json());
    }
}